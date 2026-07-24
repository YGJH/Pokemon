"""Phase 2 corpus vocab: remap real card ids to contiguous indices.

See docs/plans/corpus-mining-plan.md "Global Constraints" (Vocab bullet):
  mode="all_corpus" (locked) = every distinct card id appearing in any
  sampled episode's decks (both players), ranked by corpus frequency; remap
  to contiguous indices with PAD=0, UNKNOWN=1, real ids 2..V-1. n_vocab (if
  set) truncates to the top-N most frequent ids.
"""

from collections import Counter

from ptcg_mine.episode import deck_of

PAD = 0
UNKNOWN = 1


def build_vocab(episodes: list[dict], mode: str = "all_corpus", n_vocab: int | None = None) -> dict:
    """Build the corpus vocab from a list of parsed episodes.

    Returns a dict:
      - "id_to_index": {card_id: index}, real ids only (index >= 2)
      - "index_to_id": ["PAD", "UNKNOWN", id2, id3, ...] (index -> id, PAD/UNKNOWN as strings)
      - "freq": {card_id: corpus_count}, real ids only, in-vocab ids only
      - "size": V (= len(index_to_id))
      - "freq_coverage": [(card_id, count, cumulative_fraction), ...] in
        descending-frequency vocab order, for reporting corpus coverage.
    """
    if mode != "all_corpus":
        raise ValueError(f"build_vocab: unsupported mode {mode!r} (only 'all_corpus' is implemented)")

    counter: Counter = Counter()
    for ep in episodes:
        for p in (0, 1):
            counter.update(deck_of(ep, p))

    ordered_ids = sorted(counter.keys(), key=lambda cid: (-counter[cid], cid))
    if n_vocab is not None:
        ordered_ids = ordered_ids[:n_vocab]

    index_to_id: list = ["PAD", "UNKNOWN"] + ordered_ids
    id_to_index: dict[int, int] = {cid: idx + 2 for idx, cid in enumerate(ordered_ids)}
    freq: dict[int, int] = {cid: counter[cid] for cid in ordered_ids}

    total = sum(counter[cid] for cid in ordered_ids)
    cumulative = 0
    freq_coverage: list[tuple[int, int, float]] = []
    for cid in ordered_ids:
        cumulative += counter[cid]
        freq_coverage.append((cid, counter[cid], cumulative / total if total else 0.0))

    return {
        "id_to_index": id_to_index,
        "index_to_id": index_to_id,
        "freq": freq,
        "size": len(index_to_id),
        "freq_coverage": freq_coverage,
    }
