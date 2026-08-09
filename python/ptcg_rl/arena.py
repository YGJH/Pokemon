"""Cross-deck arena: every checkpoint against every other, decks and all.

``ptcg_rl.elo_calibrate`` rates checkpoints inside one training directory, and
it hands both sides the same pair of decks (``fixed_deck`` vs the first opponent
archetype) no matter which models are playing.  That is fine when every entrant
was trained on the same deck and wrong the moment they were not: a specialist
piloting somebody else's archetype is measured on cards it has never seen, and
the rating that comes back is mostly a statement about the deck assignment.

Here each entry carries the decklist recorded in its own ``.pt`` and takes that
deck into both seats.  What comes out is a Bradley-Terry rating decomposed into
model strength and deck strength (see :mod:`ptcg_rl.rating`), which is only
identifiable because one model — the generalist — is entered once per deck.
The generalist is the legitimate bridge: it was trained across the whole corpus
and is what every specialist warm-starts from, so it is the one set of weights
that can pilot any of these decks without being out of distribution.

Usage::

    cd python
    uv run python -m ptcg_rl.arena --games 100
    uv run python -m ptcg_rl.arena --games 20 --skip-steps      # quick pass
    uv run python -m ptcg_rl.arena --fit-only                   # re-fit, no games

Results go to ``arena_ratings.json`` — the full result matrix *and* the fit, so
the fit can be re-run, or re-parameterised, without replaying a single game.
The file is rewritten after every pair, and ``--resume`` picks up the pairs it
already holds, because a full roster is many hours of engine time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from rich.logging import RichHandler

from ptcg_rl.rating import (
    EntrySpec,
    Match,
    RatingError,
    RatingTable,
    fit_bradley_terry,
    format_table,
)

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(show_time=False)],
)
logger = logging.getLogger(__name__)

FIXED_DECK_KEY = "FIXED"
OUTPUT_NAME = "arena_ratings.json"
DEFAULT_ANCHOR = "generalist/best"


# ── Checkpoint discovery ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    """One ``.pt`` on disk, with the deck record it labels itself with."""

    name: str
    path: Path
    deck: tuple[int, ...]
    deck_key: str
    archetype_self: int | None
    vocab_sha1: str
    archetypes_sha1: str
    file_sha1: str


def dir_label(path: Path) -> str:
    """``checkpoints_a1_mcts`` → ``a1_mcts``; bare ``checkpoints`` → ``default``."""
    name = path.name
    if name == "checkpoints":
        return "default"
    return name[len("checkpoints_"):] if name.startswith("checkpoints_") else name


def ckpt_label(path: Path) -> str:
    """``ckpt-mcts-champion-001600.pt`` → ``champ-001600``."""
    stem = path.stem
    for prefix, replacement in (
        ("ckpt-mcts-champion-", "champ-"),
        ("ckpt-mcts-", "mcts-"),
        ("ckpt-", ""),
    ):
        if stem.startswith(prefix):
            return replacement + stem[len(prefix):]
    return stem


def deck_key_of(archetype_self: int | None) -> str:
    return FIXED_DECK_KEY if archetype_self is None else f"a{archetype_self}"


def read_candidate(path: Path) -> Candidate:
    """Read one checkpoint's identity.  Raises rather than guessing.

    A checkpoint without a deck record cannot enter: card ids are vocab indices,
    so handing it a decklist we picked ourselves produces a policy playing cards
    it never saw, and nothing about that failure is visible in a win rate.
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    record = ckpt.get("deck")
    if not isinstance(record, dict) or not record.get("deck"):
        raise ValueError(
            f"{path} carries no deck record — it cannot be entered in the "
            "arena, because the deck it plays is exactly what the rating is "
            "trying to separate out. Retrain, or drop it with --exclude."
        )
    if not ckpt.get("config"):
        raise ValueError(
            f"{path} carries no config record — the architecture cannot be "
            "inferred, and a wrong n_opp_arch loads silently as a plausible "
            "model with randomly-initialised belief heads."
        )
    return Candidate(
        name=f"{dir_label(path.parent)}/{ckpt_label(path)}",
        path=path,
        deck=tuple(int(c) for c in record["deck"]),
        deck_key=deck_key_of(record.get("archetype_self")),
        archetype_self=record.get("archetype_self"),
        vocab_sha1=str(record.get("vocab_sha1", "")),
        archetypes_sha1=str(record.get("archetypes_sha1", "")),
        file_sha1=hashlib.sha1(path.read_bytes()).hexdigest(),
    )


_STEP_RE = re.compile(r"^ckpt-step-\d+\.pt$")
_CHAMP_RE = re.compile(r"^ckpt-mcts-champion-(\d+)\.pt$")


def discover(
    root: Path,
    *,
    dirs: Sequence[str] | None = None,
    skip_steps: bool = False,
    exclude: Sequence[str] = (),
) -> list[Candidate]:
    """Every ``.pt`` under ``root/checkpoints*/``, deduplicated by content.

    ``checkpoints/`` and ``checkpoints_generalist/`` overlap — ``ckpt-best.pt``
    and ``ckpt-last.pt`` are byte-identical between them — and entering the same
    weights twice under two names does not just waste games: it puts a pair in
    the matrix whose true rating gap is exactly zero, which drags the fit toward
    calling *every* small gap zero.  Duplicates are dropped by SHA-1 of the file
    contents, so the check survives a rename or a re-copy.

    Bare ``checkpoints/`` is visited **last**, so when it duplicates a labelled
    directory it is the copy that loses its name.  ``checkpoints/`` is just
    ``ptcg_il.cli train``'s default output path and says nothing about what is
    in it; ``generalist/best`` is the name the anchor is referred to by
    everywhere else, and losing it to ``default/best`` would silently change
    which entry ``--anchor`` and ``--bridge`` resolve to.

    ``skip_steps`` trims the roster to the headline entries — ``best``/``last``
    and, per MCTS directory, only the highest-numbered champion.
    """
    candidate_dirs = (
        [root / d for d in dirs] if dirs
        else sorted((p for p in root.glob("checkpoints*") if p.is_dir()),
                    key=lambda p: (p.name == "checkpoints", p.name))
    )
    excluded = set(exclude)

    out: list[Candidate] = []
    by_sha: dict[str, Candidate] = {}
    for d in candidate_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"checkpoint directory {d} does not exist")
        files = sorted(d.glob("*.pt"))
        if skip_steps:
            champs = sorted(
                (int(m.group(1)), f)
                for f in files if (m := _CHAMP_RE.match(f.name))
            )
            keep_champ = champs[-1][1] if champs else None
            files = [
                f for f in files
                if not _STEP_RE.match(f.name)
                and (not _CHAMP_RE.match(f.name) or f == keep_champ)
            ]
        for f in files:
            cand = read_candidate(f)
            if cand.name in excluded:
                logger.info("  excluded: %s", cand.name)
                continue
            prior = by_sha.get(cand.file_sha1)
            if prior is not None:
                logger.info("  duplicate: %s is byte-identical to %s — dropped",
                            cand.name, prior.name)
                continue
            by_sha[cand.file_sha1] = cand
            out.append(cand)
    return out


def check_artifacts(candidates: Sequence[Candidate]) -> tuple[str, str]:
    """All entrants must be pinned to the same ``vocab.json``/``archetypes.json``.

    Card ids are vocab indices and archetype ids are cluster indices, both of
    which get reassigned whenever mining is re-run.  Mixing two generations in
    one arena is silently wrong, not loudly wrong — the older model's cards map
    to different cards, or to ``UNKNOWN_CARD``, and it simply plays badly.
    """
    if not candidates:
        raise ValueError("no checkpoints to check")
    vocab = {c.vocab_sha1 for c in candidates}
    arch = {c.archetypes_sha1 for c in candidates}
    if len(vocab) > 1 or len(arch) > 1:
        detail = "\n".join(
            f"    {c.name:<32} vocab={c.vocab_sha1} archetypes={c.archetypes_sha1}"
            for c in candidates
        )
        raise ValueError(
            "roster spans more than one artifact generation "
            f"(vocab_sha1 {sorted(vocab)}, archetypes_sha1 {sorted(arch)}):\n"
            f"{detail}\n"
            "Card ids are vocab indices and archetype ids are cluster indices, "
            "so the older checkpoints are playing a different game. Re-mine and "
            "retrain, or restrict the roster with --dirs."
        )
    return vocab.pop(), arch.pop()


# ── Roster ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArenaEntry:
    """A competitor: one model's weights on one deck."""

    name: str
    model: str
    deck_key: str
    deck: tuple[int, ...]
    path: Path

    @property
    def is_bridge(self) -> bool:
        return "@" in self.name

    def spec(self) -> EntrySpec:
        return EntrySpec(name=self.name, model=self.model, deck=self.deck_key)


def build_roster(
    candidates: Sequence[Candidate],
    bridge: str = DEFAULT_ANCHOR,
) -> list[ArenaEntry]:
    """Every candidate on its own deck, plus *bridge* on each of the others.

    Without the bridge entries the fit is rank-deficient — see
    :func:`ptcg_rl.rating.check_bridge`.  ``bridge`` must therefore be a model
    that is not out of distribution on decks it did not train on, which in this
    repo means the generalist and nothing else.
    """
    by_name = {c.name: c for c in candidates}
    if bridge not in by_name:
        raise ValueError(
            f"bridge model {bridge!r} is not among the discovered checkpoints "
            f"({sorted(by_name)[:6]}...). Without a model entered on more than "
            "one deck the rating cannot separate model strength from deck "
            "strength at all."
        )
    bridge_cand = by_name[bridge]
    if bridge_cand.archetype_self is not None:
        raise ValueError(
            f"bridge model {bridge!r} is a specialist (archetype "
            f"{bridge_cand.archetype_self}). A specialist has only ever seen "
            "its own deck, so entering it on five others measures how badly it "
            "handles unknown cards and calls the answer 'deck strength'. Use "
            "the generalist."
        )

    roster = [
        ArenaEntry(name=c.name, model=c.name, deck_key=c.deck_key,
                   deck=c.deck, path=c.path)
        for c in candidates
    ]
    decks: dict[str, tuple[int, ...]] = {}
    for c in candidates:
        decks.setdefault(c.deck_key, c.deck)

    for deck_key in sorted(decks, key=_deck_sort_key):
        if deck_key == bridge_cand.deck_key:
            continue
        roster.append(ArenaEntry(
            name=f"{bridge}@{deck_key}",
            model=bridge,
            deck_key=deck_key,
            deck=decks[deck_key],
            path=bridge_cand.path,
        ))
    return roster


def _deck_sort_key(key: str) -> tuple[int, int, str]:
    if key == FIXED_DECK_KEY:
        return (0, 0, key)
    m = re.match(r"^a(\d+)$", key)
    return (1, int(m.group(1)) if m else 0, key)


# ── Match driving ────────────────────────────────────────────────────────────


def play_half(
    actor_seat0: Callable[[list[dict]], list[dict]],
    actor_seat1: Callable[[list[dict]], list[dict]],
    deck_seat0: Sequence[int],
    deck_seat1: Sequence[int],
    n_games: int,
    *,
    n_envs: int = 8,
    seed: int = 0,
    env_factory: Any = None,
    stall_timeout: float = 120.0,
) -> tuple[int, int, int]:
    """Play *n_games* with the two seats fixed.  Returns ``(w0, w1, draws)``.

    ``RustVecEnv`` deals ``deck_self`` to seat 0 and ``deck_opp`` to seat 1
    regardless of ``our_player`` — ``our_player`` only chooses whose frame the
    reward is reported in (``ptcg_search/src/vec_env.rs``).  This function
    therefore pins ``our_player=0`` and asks the caller for decks *by seat*, so
    there is no configuration in which the decks and the models can drift apart.
    The seat swap lives one level up, in :func:`play_pair`, and swaps the decks
    along with the models.

    Observations are routed by ``select_player``, not by turn parity: the
    featurizer is egocentric, so a seat-1 observation handed to seat 0's actor
    is indistinguishable from a legitimate one and simply plays the wrong side.
    """
    if env_factory is None:
        from ptcg_rl.rust_vec_env import RustVecEnv
        env_factory = RustVecEnv

    w0 = w1 = draws = 0
    done = 0
    t_progress = time.perf_counter()

    def _tally(env: Any) -> None:
        """Count finished games, stopping at *n_games*.

        Up to ``n_envs - 1`` extra battles are already in flight when the target
        is reached, and Rust restarts a battle as soon as it finishes.  Taking
        the overspill would give the two halves different game counts, and since
        the halves are exactly what cancels first-player advantage, an uneven
        split leaks that advantage straight into the result.
        """
        nonlocal w0, w1, draws, done, t_progress
        for t in env.drain():
            t_progress = time.perf_counter()
            if done >= n_games:
                continue
            if t.reward > 0:
                w0 += 1
            elif t.reward < 0:
                w1 += 1
            else:
                draws += 1
            done += 1

    with env_factory(
        deck_self=[int(c) for c in deck_seat0],
        deck_opp=[int(c) for c in deck_seat1],
        n_envs=min(n_envs, max(1, n_games)),
        our_player=0,
        seed=seed,
    ) as env:
        while done < n_games:
            if time.perf_counter() - t_progress > stall_timeout:
                raise RuntimeError(
                    f"arena stalled: {done}/{n_games} games in "
                    f"{stall_timeout:.0f}s with no progress. A partial result "
                    "would be scored as though the missing games never "
                    "existed, which biases the pair toward whoever finished."
                )
            pending = env.poll()
            if not pending:
                _tally(env)
                time.sleep(0.001)
                continue
            t_progress = time.perf_counter()

            idx0: list[int] = []
            idx1: list[int] = []
            reqs0: list[dict] = []
            reqs1: list[dict] = []
            for pi, p in enumerate(pending):
                obs = json.loads(p["obs_json"])
                obs["search_begin_input"] = p.get("sbi", "")
                if int(p["select_player"]) == 0:
                    idx0.append(pi)
                    reqs0.append({"obs": obs, "actor": "seat0"})
                else:
                    idx1.append(pi)
                    reqs1.append({"obs": obs, "actor": "seat1"})

            replies: list[dict | None] = [None] * len(pending)
            for group, actor, idx in ((reqs0, actor_seat0, idx0),
                                      (reqs1, actor_seat1, idx1)):
                if not group:
                    continue
                out = actor(group)
                if len(out) != len(group):
                    raise RuntimeError(
                        f"actor returned {len(out)} replies for {len(group)} "
                        "requests — vec_env_reply consumes picks positionally, "
                        "so a short list applies every later pick to the wrong "
                        "battle"
                    )
                for k, rep in enumerate(out):
                    replies[idx[k]] = rep

            missing = [i for i, r in enumerate(replies) if r is None]
            if missing:
                raise RuntimeError(f"{len(missing)} unanswered request(s)")
            env.reply([r["picks"] for r in replies])  # type: ignore[index]
            _tally(env)

    return w0, w1, draws


def play_pair(
    entry_a: ArenaEntry,
    entry_b: ArenaEntry,
    actor_a: Callable[[list[dict]], list[dict]],
    actor_b: Callable[[list[dict]], list[dict]],
    games: int,
    *,
    n_envs: int = 8,
    seed: int = 0,
    env_factory: Any = None,
) -> Match:
    """``games`` split evenly: half with *a* moving first, half with *b*.

    Both halves keep each entry on its own decklist — the seat swap is there to
    cancel first-player advantage and nothing else.  Swapping seats while
    holding the deck arguments fixed would hand each policy the other's
    archetype, which is a deck swap wearing a seat swap's clothes.
    """
    per_half = max(1, games // 2)
    a_wins = b_wins = draws = 0

    for half, (first, second, actor_first, actor_second) in enumerate((
        (entry_a, entry_b, actor_a, actor_b),
        (entry_b, entry_a, actor_b, actor_a),
    )):
        w_first, w_second, d = play_half(
            actor_first, actor_second, first.deck, second.deck, per_half,
            n_envs=n_envs, seed=seed + half, env_factory=env_factory,
        )
        if first is entry_a:
            a_wins += w_first
            b_wins += w_second
        else:
            b_wins += w_first
            a_wins += w_second
        draws += d

    return Match(a=entry_a.name, b=entry_b.name,
                 wins_a=a_wins, wins_b=b_wins, draws=draws)


# ── Model loading ────────────────────────────────────────────────────────────


class ModelCache:
    """Loads policies on demand, keeping the *n* most recently used on CPU.

    Checkpoints are 216 MB on disk but only ~60 MB of parameters, so caching is
    about the ``torch.load`` cost rather than the file size.  A cache at least as
    large as the number of distinct checkpoints removes reloads entirely.
    """

    _NOT_LOADED = object()

    def __init__(self, data_dir: Path, device: Any, capacity: int = 16):
        self.data_dir = Path(data_dir)
        self.device = device
        self.capacity = max(1, capacity)
        self._cache: dict[Path, Any] = {}
        self._order: list[Path] = []
        self._all_card_feat: Any = ModelCache._NOT_LOADED
        self.n_loads = 0

    def _card_feat(self) -> Any:
        """The ``[n_cards, F_CARD]`` matrix the belief heads index into.

        Cached behind a sentinel rather than ``None``: the cached value is a
        tensor, and ``self._all_card_feat or None`` raises on one with more than
        one element.
        """
        if self._all_card_feat is not self._NOT_LOADED:
            return self._all_card_feat

        import torch
        from ptcg_il.model.cards import F_CARD

        path = self.data_dir / "engine_card_features.npy"
        if not path.exists():
            self._all_card_feat = None
            return None
        ecf = np.load(path, allow_pickle=True).item()
        max_id = max(ecf.keys()) if ecf else 0
        feat = torch.zeros(max_id + 1, F_CARD)
        for cid, row in ecf.items():
            feat[int(cid)] = torch.from_numpy(np.asarray(row, dtype=np.float32))
        self._all_card_feat = feat
        return feat

    def get(self, path: Path) -> Any:
        path = Path(path)
        if path in self._cache:
            self._order.remove(path)
            self._order.append(path)
            return self._cache[path]

        import torch
        from ptcg_il.model.policy import load_policy_state, policy_from_config

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        policy = policy_from_config(ckpt["config"], all_card_feat=self._card_feat())
        load_policy_state(policy, ckpt["model_state_dict"])
        policy.eval()
        self.n_loads += 1

        self._cache[path] = policy
        self._order.append(path)
        while len(self._order) > self.capacity:
            self._cache.pop(self._order.pop(0), None)
        return policy


# ── Persistence ──────────────────────────────────────────────────────────────


def _pair_key(a: str, b: str) -> str:
    return "␟".join(sorted((a, b)))


def load_matches(path: Path) -> dict[str, Match]:
    """Matches already on disk, keyed by unordered pair."""
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    out: dict[str, Match] = {}
    for m in doc.get("matches", []):
        match = Match(a=m["a"], b=m["b"], wins_a=int(m["wins_a"]),
                      wins_b=int(m["wins_b"]), draws=int(m.get("draws", 0)))
        out[_pair_key(match.a, match.b)] = match
    return out


def write_results(
    path: Path,
    roster: Sequence[ArenaEntry],
    matches: Iterable[Match],
    meta: dict,
    table: RatingTable | None = None,
) -> Path:
    """Write the result matrix (and the fit, if it ran) to *path*."""
    doc: dict[str, Any] = dict(meta)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc["entries"] = [
        {"name": e.name, "model": e.model, "deck": e.deck_key,
         "checkpoint": str(e.path), "bridge": e.is_bridge}
        for e in roster
    ]
    doc["matches"] = [
        {"a": m.a, "b": m.b, "wins_a": m.wins_a, "wins_b": m.wins_b,
         "draws": m.draws, "n": m.n}
        for m in matches
    ]
    doc["ratings"] = table.as_dict() if table is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path


def fit_from_document(doc: dict) -> RatingTable:
    """Re-run the fit over a saved ``arena_ratings.json`` — no games needed."""
    entries = [
        EntrySpec(name=e["name"], model=e["model"], deck=e["deck"])
        for e in doc["entries"]
    ]
    matches = [
        Match(a=m["a"], b=m["b"], wins_a=int(m["wins_a"]),
              wins_b=int(m["wins_b"]), draws=int(m.get("draws", 0)))
        for m in doc["matches"]
    ]
    return fit_bradley_terry(matches, entries, doc.get("anchor", DEFAULT_ANCHOR))


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptcg_rl.arena",
        description="Cross-deck round robin + Bradley-Terry ratings with an "
                    "additive deck term",
    )
    p.add_argument("--root", type=str, default=".",
                   help="directory holding the checkpoints*/ directories")
    p.add_argument("--dirs", type=str, nargs="*", default=None,
                   help="explicit checkpoint directories (default: checkpoints*)")
    p.add_argument("--exclude", type=str, nargs="*", default=[],
                   help="entry names to leave out, e.g. default/step-0002000")
    p.add_argument("--skip-steps", action="store_true",
                   help="headline entries only: best/last plus the latest MCTS "
                        "champion per directory")
    p.add_argument("--games", type=int, default=100,
                   help="games per pair, split evenly across the two seats")
    p.add_argument("--anchor", type=str, default=DEFAULT_ANCHOR,
                   help="entry pinned to 1500")
    p.add_argument("--bridge", type=str, default=DEFAULT_ANCHOR,
                   help="model entered once per deck; this is what makes model "
                        "strength and deck strength separable")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out", type=str, default=OUTPUT_NAME)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--model-cache", type=int, default=16,
                   help="policies kept in RAM; >= the number of distinct "
                        "checkpoints avoids all reloads (~60 MB each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", action="store_true",
                   help="skip pairs already recorded in --out")
    p.add_argument("--fit-only", action="store_true",
                   help="re-fit from --out without playing any games")
    p.add_argument("--dry-run", action="store_true",
                   help="print the roster and the cost estimate, then stop")
    return p


def _fmt_dur(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_path = Path(args.out)

    if args.fit_only:
        if not out_path.exists():
            logger.error("%s does not exist — nothing to fit", out_path)
            return 1
        doc = json.loads(out_path.read_text())
        try:
            table = fit_from_document(doc)
        except RatingError as exc:
            logger.error("fit refused: %s", exc)
            return 2
        print(format_table(table))
        # Rewrite the fit in place.  ``write_results`` rebuilds ``entries`` and
        # ``matches`` from its arguments, so it must not be handed the empty
        # lists here — that would delete the games this fit was made of.
        doc["ratings"] = table.as_dict()
        doc["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        return 0

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(args.root)
    data_dir = Path(args.data_dir)

    logger.info("Discovering checkpoints under %s", root.resolve())
    candidates = discover(root, dirs=args.dirs, skip_steps=args.skip_steps,
                          exclude=args.exclude)
    if not candidates:
        logger.error("no checkpoints found")
        return 1
    vocab_sha1, archetypes_sha1 = check_artifacts(candidates)

    roster = build_roster(candidates, bridge=args.bridge)
    if not any(e.name == args.anchor for e in roster):
        logger.error("anchor %r is not in the roster", args.anchor)
        return 1

    n = len(roster)
    n_pairs = n * (n - 1) // 2
    decks = sorted({e.deck_key for e in roster}, key=_deck_sort_key)
    logger.info("Roster: %d entries over %d decks → %d pairs × %d games = %d games",
                n, len(decks), n_pairs, args.games, n_pairs * args.games)
    print()
    for e in sorted(roster, key=lambda e: (_deck_sort_key(e.deck_key), e.name)):
        tag = "  ← bridge" if e.is_bridge else ("  ← anchor" if e.name == args.anchor else "")
        print(f"  {e.name:<40} {e.deck_key:<6}{tag}")
    print()

    if args.dry_run:
        return 0

    # ── Shared artifacts ────────────────────────────────────────────────
    from ptcg_il.featurizer import normalize_vocab

    with open(data_dir / "vocab.json") as f:
        vocab = normalize_vocab(json.load(f))
    ecf_path = data_dir / "engine_card_features.npy"
    eaf_path = data_dir / "engine_attack_features.npy"
    engine_card_features = (
        np.load(ecf_path, allow_pickle=True).item() if ecf_path.exists() else None
    )
    engine_attack_features = (
        np.load(eaf_path, allow_pickle=True).item() if eaf_path.exists() else None
    )
    evo_path = data_dir / "evolution_map.npy"
    evolution_map = (
        np.load(evo_path, allow_pickle=True).item() if evo_path.exists() else None
    )
    if engine_card_features is None:
        logger.warning("no engine_card_features.npy in %s — every card gets zero "
                       "features and the whole arena measures noise", data_dir)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "games_per_pair": args.games,
        "seed": args.seed,
        "anchor": args.anchor,
        "bridge": args.bridge,
        "vocab_sha1": vocab_sha1,
        "archetypes_sha1": archetypes_sha1,
        "data_dir": str(data_dir),
    }

    played: dict[str, Match] = load_matches(out_path) if args.resume else {}
    if played:
        logger.info("Resuming: %d pair(s) already recorded in %s",
                    len(played), out_path)

    cache = ModelCache(data_dir, device, capacity=args.model_cache)
    from ptcg_rl.rollout import PolicyActor

    def actor_for(entry: ArenaEntry, seed: int) -> Callable[[list[dict]], list[dict]]:
        policy = cache.get(entry.path).to(device)
        return PolicyActor(
            policy, vocab, device=str(device), greedy=True, seed=seed,
            engine_card_features=engine_card_features,
            engine_attack_features=engine_attack_features,
            evolution_map=evolution_map,
        )

    t_start = time.perf_counter()
    pair_idx = 0
    n_new = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_idx += 1
            a, b = roster[i], roster[j]
            key = _pair_key(a.name, b.name)
            if key in played:
                continue

            pair_seed = args.seed + pair_idx * 1009
            match = play_pair(
                a, b, actor_for(a, pair_seed), actor_for(b, pair_seed + 1),
                args.games, n_envs=args.n_envs, seed=pair_seed,
            )
            played[key] = match
            n_new += 1

            # Rate is per *played* pair — dividing by pair_idx would count the
            # pairs --resume skipped for free and report an ETA of nearly zero.
            elapsed = time.perf_counter() - t_start
            eta = elapsed / n_new * (n_pairs - len(played))
            logger.info(
                "[%d/%d] %s vs %s: %d-%d-%d (wr=%.3f) · elapsed %s · eta %s",
                pair_idx, n_pairs, a.name, b.name,
                match.wins_a, match.wins_b, match.draws,
                match.score_a / max(match.n, 1),
                _fmt_dur(elapsed), _fmt_dur(eta),
            )
            write_results(out_path, roster, played.values(), meta)

    table = None
    try:
        table = fit_bradley_terry(
            list(played.values()), [e.spec() for e in roster], args.anchor,
        )
    except RatingError as exc:
        logger.error("fit refused: %s", exc)
    write_results(out_path, roster, played.values(), meta, table)

    if table is None:
        logger.error("results written to %s; re-run with --fit-only after "
                     "fixing the roster", out_path)
        return 2

    print(format_table(table))
    print(f"\nsaved to {out_path}  ·  {_fmt_dur(time.perf_counter() - t_start)}"
          f"  ·  {cache.n_loads} checkpoint load(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
