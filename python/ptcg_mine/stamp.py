"""Stage-freshness stamps for the pipeline's two expensive, deterministic stages.

Phase 2 (mine) re-parses every episode under ``raw/`` — ~15 min over the real
9910-episode corpus — and Phase 3 (build-shards) re-featurizes all of it —
~35 min.  Both are pure functions of (raw corpus, config knobs, the code that
implements them), so re-running them on an unchanged corpus produces
byte-identical artifacts and costs ~50 min per pipeline invocation.

This module records a fingerprint of those inputs after a stage succeeds, and
answers "is the stamp still valid?" before the next run.

A bare existence check would not be safe here.  Card ids are vocab indices and
archetype ids are cluster indices, so artifacts that merely *exist* can be
silently mislabelled relative to the code or corpus that would rebuild them —
CLAUDE.md records exactly this failure mode for the ``opt_card_id`` featurizer
fix.  The fingerprint therefore covers four things:

  * **raw corpus** — sorted (relpath, size) of every ``*.json`` under raw_dir.
  * **config** — the knobs the pipeline passed on the command line.
  * **code** — the source files that actually decide the stage's output.
    Deliberately *not* the whole package: ``ptcg_il/cli.py`` gains train-only
    flags constantly and must not cost a 35-minute rebuild.
  * **upstream artifacts** — content hashes of the files the stage consumes
    from an earlier stage.  This is what chains the stages: a new vocab.json
    from mine invalidates the shards built against the old one.

Outputs are checked for existence separately, so deleting ``data/shards/``
forces a rebuild regardless of the fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# `check` uses a distinct exit code for "stale" so the shell can tell a
# needs-rebuild answer apart from the module itself failing.
EXIT_STALE = 10


@dataclass(frozen=True)
class StageSpec:
    """What a stage reads, writes, and is implemented by."""

    #: Source files (relative to source_root) whose content changes the output.
    code: tuple[str, ...]
    #: Paths under data_dir that must exist for the stamp to count as valid.
    #: A trailing "/" means "directory that must contain at least one file".
    outputs: tuple[str, ...]
    #: Files under data_dir produced by an *earlier* stage, hashed by content.
    upstream: tuple[str, ...] = field(default=())


_MINE_CODE = (
    "ptcg_mine/mine.py",
    "ptcg_mine/episode.py",
    "ptcg_mine/stats.py",
    "ptcg_mine/archetype.py",
    "ptcg_mine/vocab.py",
    "ptcg_mine/cards.py",
    "ptcg_mine/artifacts.py",
    "ptcg_mine/config.py",
)

STAGES: dict[str, StageSpec] = {
    "mine": StageSpec(
        code=_MINE_CODE,
        outputs=(
            "vocab.json",
            "archetypes.json",
            "card_static_table.npy",
            "attack_static_table.npy",
            # Written by the same Phase-2 block and consumed by build-shards;
            # a mine stamp that passes without them lets the next stage read a
            # table left over from a previous run.
            "engine_card_features.npy",
            "engine_attack_features.npy",
        ),
    ),
    "shards": StageSpec(
        # Selection (episode/stats/archetype) decides which games are kept;
        # featurizer/ref_map/belief_labels decide what each sample contains.
        code=(
            "ptcg_il/shard_writer.py",
            "ptcg_il/featurizer.py",
            "ptcg_il/ref_map.py",
            "ptcg_il/belief_labels.py",
            "ptcg_mine/episode.py",
            "ptcg_mine/stats.py",
            "ptcg_mine/archetype.py",
            "ptcg_mine/config.py",
        ),
        outputs=("meta.parquet", "shards/"),
        # The engine feature tables belong here even though no file in `code`
        # builds them: `shard_writer` loads them and the featurizer bakes their
        # *contents* into every `*_card_feat` tensor it writes.  Without them a
        # `ptcg_mine/cards.py` edit invalidates the mine stamp, rewrites the
        # tables, and leaves this stage reporting itself cached -- so 20 GB of
        # shards keep features derived from the old table and nothing raises.
        # These are the `engine_*` files, not `*_static_table.npy`: the latter
        # are built off the mined vocab and no longer feed the featurizer.
        upstream=(
            "vocab.json",
            "archetypes.json",
            "engine_card_features.npy",
            "engine_attack_features.npy",
        ),
    ),
}


#: Config knobs each stage's output depends on, as (stamp key, config attr).
#: Only knobs a caller can actually vary — the pipeline and both CLIs read the
#: values off `MineConfig`, so there is one source of truth rather than a list
#: duplicated in shell.
STAGE_PARAMS: dict[str, tuple[tuple[str, str], ...]] = {
    "mine": (
        ("seed", "seed"),
        ("k-experts", "k_experts"),
        ("g-min", "g_min"),
        ("jaccard-thresh", "jaccard_thresh"),
        ("n-self", "n_self"),
        ("n-opp", "n_opp"),
        ("vocab-mode", "vocab_mode"),
        ("n-vocab", "n_vocab"),
    ),
    "shards": (
        ("k-experts", "k_experts"),
        ("g-min", "g_min"),
        ("jaccard-thresh", "jaccard_thresh"),
        ("h-max", "h_max"),
        ("o-max", "o_max"),
        ("d-max", "d_max"),
    ),
}


def params_from_config(stage: str, config, **extra) -> dict[str, str]:
    """The param dict for `stage`, read off a MineConfig."""
    params = {
        key: getattr(config, attr)
        for key, attr in STAGE_PARAMS[stage]
        if hasattr(config, attr)
    }
    params.update(extra)
    return {k: str(v) for k, v in params.items()}


def _canon(value: str) -> str:
    """Canonicalise a param value so equal numbers compare equal.

    ``--jaccard-thresh 0.90`` and ``0.9`` describe the same run; without this a
    reformatted invocation costs a 35-minute rebuild for nothing.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return repr(int(f)) if f.is_integer() else repr(f)


def _canon_params(params: dict) -> dict[str, str]:
    return {str(k): _canon(v) for k, v in params.items()}


def all_code_files() -> tuple[str, ...]:
    """Every source file referenced by any stage (used by tests to build a
    fake source tree)."""
    seen: dict[str, None] = {}
    for spec in STAGES.values():
        for rel in spec.code:
            seen[rel] = None
    return tuple(seen)


def _sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_fingerprint(raw_dir: Path) -> dict:
    """Hash the corpus by (relpath, size) rather than content: hashing 41 GB
    on every pipeline start would cost more than the stage it protects, and a
    downloaded episode that changes at all changes its byte count in practice.

    A missing raw_dir yields n_files=0, which never matches a real stamp — so
    a wrong --raw-dir re-runs the stage instead of trusting the cache.
    """
    raw_dir = Path(raw_dir)
    h = hashlib.sha1()
    n = 0
    if raw_dir.is_dir():
        for path in sorted(raw_dir.rglob("*.json")):
            h.update(str(path.relative_to(raw_dir)).encode())
            h.update(b"\0")
            h.update(str(path.stat().st_size).encode())
            h.update(b"\n")
            n += 1
    return {"n_files": n, "sha1": h.hexdigest()}


def code_fingerprint(spec: StageSpec, source_root: Path | None = None) -> dict[str, str]:
    """Content hash per source file. A file that does not exist maps to
    "missing" rather than raising, so a renamed module shows up as a change."""
    source_root = Path(source_root) if source_root is not None else _default_source_root()
    out: dict[str, str] = {}
    for rel in spec.code:
        p = source_root / rel
        out[rel] = _sha1_of_file(p) if p.is_file() else "missing"
    return out


def upstream_fingerprint(spec: StageSpec, data_dir: Path) -> dict[str, str]:
    data_dir = Path(data_dir)
    out: dict[str, str] = {}
    for rel in spec.upstream:
        p = data_dir / rel
        out[rel] = _sha1_of_file(p) if p.is_file() else "missing"
    return out


def missing_outputs(spec: StageSpec, data_dir: Path) -> list[str]:
    data_dir = Path(data_dir)
    missing = []
    for rel in spec.outputs:
        p = data_dir / rel.rstrip("/")
        if rel.endswith("/"):
            if not p.is_dir() or not any(p.iterdir()):
                missing.append(rel)
        elif not p.is_file():
            missing.append(rel)
    return missing


def stamp_path(stage: str, data_dir: Path) -> Path:
    return Path(data_dir) / f".stamp-{stage}.json"


def compute(
    stage: str,
    *,
    raw_dir: Path,
    data_dir: Path,
    params: dict[str, str],
    source_root: Path | None = None,
) -> dict:
    """The fingerprint record for `stage` as the inputs stand right now."""
    spec = STAGES[stage]
    return {
        "stage": stage,
        "raw": raw_fingerprint(raw_dir),
        "params": {str(k): str(v) for k, v in sorted(params.items())},
        "code": code_fingerprint(spec, source_root),
        "upstream": upstream_fingerprint(spec, data_dir),
    }


def read(stage: str, data_dir: Path) -> dict | None:
    """The recorded stamp for `stage`, or None if absent/unreadable."""
    path = stamp_path(stage, data_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def check(
    stage: str,
    *,
    raw_dir: Path,
    data_dir: Path,
    params: dict[str, str],
    source_root: Path | None = None,
    require_summary: bool = False,
) -> tuple[bool, str]:
    """Return (can_skip, reason). `reason` names the first input that differs
    so callers can say *why* they are spending 35 minutes.

    `require_summary` is for callers that reprint the previous run's summary
    from the stamp: a stamp written before summaries existed is treated as
    stale rather than reported with fields missing.
    """
    spec = STAGES[stage]  # KeyError on an unknown stage: a typo must not skip work

    missing = missing_outputs(spec, data_dir)
    if missing:
        return False, f"missing output: {', '.join(missing)}"

    path = stamp_path(stage, data_dir)
    if not path.is_file():
        return False, "no stamp file"
    try:
        old = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False, "unreadable stamp file"

    new = compute(stage, raw_dir=raw_dir, data_dir=data_dir,
                  params=params, source_root=source_root)

    if old.get("raw") != new["raw"]:
        o, n = old.get("raw") or {}, new["raw"]
        return False, (f"raw corpus changed "
                       f"({o.get('n_files', '?')} → {n['n_files']} episodes)")
    old_params = _canon_params(old.get("params") or {})
    new_params = _canon_params(new["params"])
    if old_params != new_params:
        return False, f"config changed: {_diff_keys(old_params, new_params)}"
    if old.get("code") != new["code"]:
        changed = _diff_keys(old.get("code") or {}, new["code"])
        return False, f"code changed: {changed}"
    if old.get("upstream") != new["upstream"]:
        changed = _diff_keys(old.get("upstream") or {}, new["upstream"])
        return False, f"upstream artifact changed: {changed}"
    if require_summary and not old.get("summary"):
        return False, "stamp predates summary recording"
    return True, "fingerprint match"


def _diff_keys(old: dict, new: dict) -> str:
    keys = sorted(set(old) | set(new))
    return ", ".join(k for k in keys if old.get(k) != new.get(k)) or "(reordered)"


def write(
    stage: str,
    *,
    raw_dir: Path,
    data_dir: Path,
    params: dict[str, str],
    source_root: Path | None = None,
    summary: dict | None = None,
) -> Path:
    """Record the current fingerprint. Call only after the stage succeeded.

    `summary` is the stage's own result dict, kept so a later cached run can
    report the same numbers instead of printing nothing.
    """
    rec = compute(stage, raw_dir=raw_dir, data_dir=data_dir,
                  params=params, source_root=source_root)
    if summary is not None:
        rec["summary"] = summary
    rec["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = stamp_path(stage, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return path


# ---- CLI ----------------------------------------------------------------

def _default_source_root() -> Path:
    # ptcg_mine/stamp.py → python/
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptcg_mine.stamp",
        description="Check or record whether an expensive pipeline stage needs to re-run.",
    )
    p.add_argument("action", choices=["check", "write"])
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--raw-dir", default="raw")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--source-root", default=None,
                   help="Package root to hash stage code from (default: the "
                        "python/ directory this module lives in)")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="A config knob the stage was invoked with; repeatable. "
                        "Any change re-runs the stage.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params: dict[str, str] = {}
    for item in args.param:
        key, _, value = item.partition("=")
        params[key] = value
    source_root = Path(args.source_root) if args.source_root else _default_source_root()
    kw = dict(raw_dir=Path(args.raw_dir), data_dir=Path(args.data_dir),
              params=params, source_root=source_root)

    if args.action == "write":
        print(f"stamped {write(args.stage, **kw)}")
        return 0

    ok, reason = check(args.stage, **kw)
    if ok:
        print(f"cached ({reason})")
        return 0
    print(f"stale ({reason})")
    return EXIT_STALE


if __name__ == "__main__":
    sys.exit(main())
