"""Parsing and validation for Kaggle Pokemon TCG episode replay JSON.

Episode format (see docs/plans/corpus-mining-plan.md "Global Constraints"):
  - ep["info"]["TeamNames"] = [team0, team1]
  - ep["rewards"] = [r0, r1] with +1 win / -1 loss / 0 draw
  - ep["steps"] is list[[recordP0, recordP1]]
  - deck_of(ep, p) = ep["steps"][1][p]["action"], a 60-int decklist
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Iterator


def project_for_selection(ep: dict) -> dict:
    """Reduce an episode to the fields the selection stages actually read.

    Expert selection, archetype clustering, vocab building and the kept-game
    filter only ever touch TeamNames, rewards, statuses and the two opening
    deck actions at ``steps[1][p]``. Everything else in ``steps`` is the
    turn-by-turn game log: ~4 MB on disk per episode, ~14.5 MB once parsed
    into Python objects. Retaining whole episodes therefore costs ~145 GB at
    the 10k-episode target, which the miner cannot survive.

    The projection is ~1 KB, so a caller holding one per episode scales with
    episode *count*, not corpus bytes. ``steps`` keeps its two-element shape so
    ``validate_episode``'s ``len(steps) >= 2`` check and ``deck_of``'s
    ``steps[1][p]["action"]`` indexing keep working against the projection.

    Stages that need the real game log (Phase 3 featurization) must re-read the
    episode from disk rather than expect it here.
    """
    return {
        "info": ep.get("info", {}),
        "statuses": ep.get("statuses"),
        "rewards": ep.get("rewards"),
        "steps": _slim_steps(ep.get("steps") or []),
    }


def load_episode(path: str | Path) -> dict:
    """Load an episode JSON file into a dict."""
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# Partial parse: read the projection without decoding the game log
# ============================================================
#
# `project_for_selection` throws away ~98.6% of what `json.load` just built.
# Everything the selection stages read lives in the head of the document --
# measured on the real corpus, `info` starts at byte 188, `rewards` at 433,
# `statuses` at 1565 and `steps` at 1595 of a 6.32 MB episode, and `steps[0]`
# and `steps[1]` are the first two of ~210 entries.  Decoding the whole file to
# reach them costs 81 MB/s over 65 GB, i.e. ~13 min per pass.
#
# The reader below walks the top-level object with `raw_decode`, decodes only
# `steps[0..1]`, and stops as soon as all four keys are in hand.  It is not a
# hand-rolled JSON parser: every *value* is still decoded by the stdlib, so
# strings containing braces, escapes and the like are handled by the same code
# as before.  Only the object/array framing is walked here.

_DECODER = json.JSONDecoder()
_WS = " \t\n\r"
_SELECTION_KEYS = frozenset({"info", "statuses", "rewards", "steps"})


class ProjectionIncomplete(Exception):
    """The head of the document did not contain every selection key.

    Raised when a key the projection needs sits *after* `steps` (or is absent),
    which the partial reader cannot reach without decoding the game log.  The
    caller falls back to a full parse; correctness never depends on the layout.
    """


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in _WS:
        i += 1
    return i


def _at(s: str, i: int) -> str:
    if i >= len(s):
        raise json.JSONDecodeError("unexpected end of episode", s, max(i - 1, 0))
    return s[i]


def _scan_steps_head(s: str, i: int) -> tuple[list, int, bool]:
    """Decode at most the first two elements of the array at *i*.

    Returns ``(elements, next_index, truncated)``.  When *truncated* is True the
    array's remaining elements were never read and *next_index* is meaningless,
    so the caller must stop scanning the enclosing object.
    """
    if _at(s, i) != "[":
        raise json.JSONDecodeError("expected 'steps' to be an array", s, i)
    i = _skip_ws(s, i + 1)
    head: list = []
    if _at(s, i) == "]":
        return head, i + 1, False
    while True:
        value, i = _DECODER.raw_decode(s, i)
        head.append(value)
        i = _skip_ws(s, i)
        char = _at(s, i)
        if char == "]":
            return head, i + 1, False
        if char != ",":
            raise json.JSONDecodeError("expected ',' or ']' in 'steps'", s, i)
        if len(head) == 2:
            return head, i, True
        i = _skip_ws(s, i + 1)


def _slim_steps(head: list) -> list:
    """Reduce decoded ``steps[0..1]`` to the shape `project_for_selection` emits.

    Short or misshapen input is passed through rather than repaired, so
    `validate_episode` rejects it exactly as it would have rejected the full
    episode.
    """
    if len(head) < 2:
        return list(head)
    second = head[1]
    try:
        slim = [{"action": rec["action"]} for rec in second]
    except (TypeError, KeyError, IndexError):
        return [[], second]
    return [[], slim]


def parse_projection(text: str) -> dict:
    """Build the selection projection from *text* without decoding the game log.

    Raises `ProjectionIncomplete` if the document's key order puts a needed key
    beyond the point where scanning stops, and `json.JSONDecodeError` if the
    head of the document is malformed.
    """
    i = _skip_ws(text, 0)
    if _at(text, i) != "{":
        raise json.JSONDecodeError("episode is not a JSON object", text, i)
    i = _skip_ws(text, i + 1)

    found: dict = {}
    if _at(text, i) != "}":
        while True:
            key, i = _DECODER.raw_decode(text, i)
            if not isinstance(key, str):
                raise json.JSONDecodeError("expected an object key", text, i)
            i = _skip_ws(text, i)
            if _at(text, i) != ":":
                raise json.JSONDecodeError("expected ':' after key", text, i)
            i = _skip_ws(text, i + 1)

            if key == "steps":
                head, i, truncated = _scan_steps_head(text, i)
                found["steps"] = head
                if truncated:
                    break
            else:
                value, i = _DECODER.raw_decode(text, i)
                if key in _SELECTION_KEYS:
                    found[key] = value

            if len(found) == len(_SELECTION_KEYS):
                break

            i = _skip_ws(text, i)
            char = _at(text, i)
            if char == "}":
                break
            if char != ",":
                raise json.JSONDecodeError("expected ',' or '}'", text, i)
            i = _skip_ws(text, i + 1)

    if len(found) != len(_SELECTION_KEYS):
        raise ProjectionIncomplete(
            f"missing {sorted(_SELECTION_KEYS - found.keys())} in the document head"
        )

    return {
        "info": found["info"],
        "statuses": found["statuses"],
        "rewards": found["rewards"],
        "steps": _slim_steps(found["steps"]),
    }


def load_projection(path: str | Path) -> dict:
    """Load only the fields the selection stages read from the episode at *path*.

    Equivalent to ``project_for_selection(load_episode(path))``, and falls back
    to exactly that when the partial reader cannot complete.  Raises the same
    `json.JSONDecodeError` / `OSError` as `load_episode` for unusable files.

    One deliberate difference: corruption *after* ``steps[1]`` is no longer
    detected here, because those bytes are never decoded.  Such an episode
    passes selection and then fails its full parse in the featurize pass, which
    must therefore tolerate an episode that will not load.
    """
    # Whole-file read, deliberately, even though the projection needs only a
    # prefix.  Reading incrementally was tried and is *slower*: the needed
    # prefix is a median 58% of the file (`steps[0]` and `steps[1]` are the
    # deck-selection steps, and their observations dwarf every later step), so
    # a growing-prefix loop re-parses several times and still reads most of the
    # bytes.  Text mode rather than bytes-then-decode keeps one copy resident.
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError as exc:
        raise json.JSONDecodeError(f"episode is not valid UTF-8: {exc}", "", 0) from exc
    try:
        return parse_projection(text)
    except ProjectionIncomplete:
        return project_for_selection(json.loads(text))


def teams(ep: dict) -> tuple[str, str]:
    """Return the (team0, team1) names from ep["info"]["TeamNames"]."""
    t0, t1 = ep["info"]["TeamNames"]
    return (t0, t1)


def rewards(ep: dict) -> tuple[int, int]:
    """Return the (r0, r1) final rewards from ep["rewards"]."""
    r0, r1 = ep["rewards"]
    return (r0, r1)


def deck_of(ep: dict, p: int) -> list[int]:
    """Return player p's 60-card decklist: ep["steps"][1][p]["action"].

    Raises ValueError if the extracted deck is not exactly 60 cards.
    """
    deck = ep["steps"][1][p]["action"]
    if len(deck) != 60:
        raise ValueError(
            f"deck_of(ep, {p}) expected 60 cards, got {len(deck)}"
        )
    return deck


def validate_episode(ep: dict) -> bool:
    """An episode is usable iff:
      - statuses == ["DONE", "DONE"]
      - len(steps) >= 2
      - len(rewards) == 2
      - both decks (deck_of(ep, 0), deck_of(ep, 1)) have length 60
    """
    # `len(... or [])` rather than `len(ep.get(k, []))`: the projection records a
    # missing key as an explicit None, so a validator that assumed "absent" and
    # "null" were the same thing would raise TypeError instead of rejecting.
    if ep.get("statuses") != ["DONE", "DONE"]:
        return False
    if len(ep.get("steps") or []) < 2:
        return False
    if len(ep.get("rewards") or []) != 2:
        return False
    for p in (0, 1):
        try:
            deck_of(ep, p)
        except (ValueError, KeyError, IndexError, TypeError):
            return False
    return True


# ============================================================
# Parallel selection scan
# ============================================================
#
# Both selection passes -- Phase 2's `mine.load_raw_episodes` and Phase 3's
# `shard_writer` pass A -- read every episode under raw/ and keep only the
# projection.  That is ~10 min of pure `json` decoding over the 65 GB corpus in
# one process, and it is embarrassingly parallel over files.

class ScanStatus(Enum):
    """Outcome of scanning one episode file.

    An Enum rather than plain string constants on purpose: results come back
    through `pickle` from worker processes, and an unpickled *str* is a fresh
    object, so `status is OK` silently reads False for every parallel result
    while staying True in the serial path.  Enum members unpickle to the
    canonical singleton, which makes `is` and `==` agree either way.
    """

    OK = "ok"
    UNREADABLE = "unreadable"   # the file could not be parsed at all
    INVALID = "invalid"         # it parsed but failed validate_episode


OK = ScanStatus.OK
UNREADABLE = ScanStatus.UNREADABLE
INVALID = ScanStatus.INVALID

# A pool costs ~1 s to stand up against ~40 ms per episode, so below this many
# files it is pure overhead.  Chunking amortises IPC over a batch (each result
# is ~1 KB) while leaving every worker enough chunks that one slow file cannot
# strand a core.
_PARALLEL_SCAN_MIN_FILES = 256
_SCAN_CHUNKSIZE = 8


def scan_episode(path_str: str) -> tuple[str, dict | None, ScanStatus]:
    """Pool worker: ``(episode_id, projection | None, status)``.

    Top-level (so it pickles) and returns ~1 KB — the multi-MB episode it was
    built from never crosses the process boundary.
    """
    path = Path(path_str)
    try:
        proj = load_projection(path)
    except (json.JSONDecodeError, OSError):
        return path.stem, None, UNREADABLE
    if validate_episode(proj):
        return path.stem, proj, OK
    return path.stem, None, INVALID


def scan_projections(paths, jobs: int | None = None) -> Iterator[tuple[str, dict | None, ScanStatus]]:
    """Yield ``(episode_id, projection | None, status)`` for each path, in order.

    *jobs* defaults to ``os.cpu_count()``; 1 runs in-process.  Results follow
    *paths* whatever the worker count, so nothing downstream that depends on
    corpus order — expert ranking, archetype cluster ids — can shift with it.

    Callers do their own accounting because they do not agree on it: Phase 2
    counts only files that parsed as "loaded", Phase 3 counts every file it
    opened.
    """
    paths = list(paths)
    n_jobs = (os.cpu_count() or 1) if jobs is None else max(1, jobs)

    if n_jobs <= 1 or len(paths) < _PARALLEL_SCAN_MIN_FILES:
        for path in paths:
            yield scan_episode(str(path))
        return

    import multiprocessing as mp

    # fork where it exists: this scan is read-only and runs before any engine
    # handle, CUDA context or thread pool exists, so the usual fork hazards do
    # not apply — while spawn would re-import numpy/pandas in every worker and
    # oblige every caller to carry a __main__ guard.
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - non-fork platforms
        ctx = mp.get_context()
    with ctx.Pool(processes=n_jobs) as pool:
        yield from pool.imap(scan_episode, [str(p) for p in paths],
                             chunksize=_SCAN_CHUNKSIZE)
