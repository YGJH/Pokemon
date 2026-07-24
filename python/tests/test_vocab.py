"""Tests for ptcg_mine.vocab: build_vocab (Phase 2 corpus vocab)."""

from ptcg_mine.vocab import build_vocab


def _make_ep(deck0, deck1):
    """A minimal well-formed synthetic episode (see episode.py docstring)."""
    return {
        "info": {"TeamNames": ["A", "B"]},
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [{"action": None}, {"action": None}],
            [{"action": deck0}, {"action": deck1}],
        ],
    }


_filler = iter(range(900000, 901000))


def _deck(ids_with_counts):
    """Build a 60-card deck from a list of ids (repeated as given), padded
    with unique filler ids (each appearing once) to reach exactly 60 cards
    so filler never perturbs the frequency ordering under test."""
    deck = list(ids_with_counts)
    while len(deck) < 60:
        deck.append(next(_filler))
    assert len(deck) == 60
    return deck


def test_build_vocab_pad_unknown_reserved():
    episodes = [_make_ep(_deck([1, 2, 3]), _deck([4, 5, 6]))]
    vocab = build_vocab(episodes, mode="all_corpus")
    assert vocab["index_to_id"][0] == "PAD"
    assert vocab["index_to_id"][1] == "UNKNOWN"
    assert 0 not in vocab["id_to_index"].values()
    assert 1 not in vocab["id_to_index"].values()


def test_build_vocab_descending_frequency_order():
    # id 100 appears 5x, id 200 appears 3x, id 300 appears 1x (across two decks)
    deck0 = _deck([100] * 5 + [200] * 3 + [300])
    deck1 = _deck([100] * 5 + [200] * 3 + [300])
    episodes = [_make_ep(deck0, deck1)]
    vocab = build_vocab(episodes, mode="all_corpus")
    # index 2 => highest freq, etc.
    ids_in_order = [vocab["index_to_id"][i] for i in range(2, vocab["size"])]
    assert ids_in_order[:3] == [100, 200, 300]
    assert vocab["freq"][100] == 10  # 5 per deck * 2 decks
    assert vocab["id_to_index"][100] == 2
    assert vocab["id_to_index"][200] == 3
    assert vocab["id_to_index"][300] == 4


def test_build_vocab_tie_break_by_id():
    # 20, 30, 50 all appear exactly once (tied with the unique filler ids, but
    # those are all >= 900000 so ascending-id tie-break puts 20 < 30 < 50 first).
    deck0 = _deck([50, 20, 30])
    episodes = [_make_ep(deck0, _deck([]))]
    vocab = build_vocab(episodes, mode="all_corpus")
    idx20 = vocab["id_to_index"][20]
    idx30 = vocab["id_to_index"][30]
    idx50 = vocab["id_to_index"][50]
    assert idx20 < idx30 < idx50


def test_build_vocab_all_corpus_keeps_every_distinct_id():
    deck0 = _deck([1, 2, 3])
    deck1 = _deck([4, 5, 6])
    episodes = [_make_ep(deck0, deck1)]
    vocab = build_vocab(episodes, mode="all_corpus")
    all_ids_in_decks = set(deck0) | set(deck1)
    assert set(vocab["id_to_index"].keys()) == all_ids_in_decks
    assert vocab["size"] == len(all_ids_in_decks) + 2  # + PAD, UNKNOWN


def test_build_vocab_n_vocab_truncates_to_top_n():
    deck0 = _deck([100] * 10 + [200] * 5 + [300] * 3 + [400] * 2 + [500] * 1)
    episodes = [_make_ep(deck0, _deck([]))]
    vocab = build_vocab(episodes, mode="all_corpus", n_vocab=2)
    # size = n_vocab + 2 reserved slots
    assert vocab["size"] == 4
    assert set(vocab["id_to_index"].keys()) == {100, 200}
    assert vocab["id_to_index"][100] == 2
    assert vocab["id_to_index"][200] == 3


def test_build_vocab_index_to_id_roundtrip():
    episodes = [_make_ep(_deck([7, 8, 9]), _deck([10, 11, 12]))]
    vocab = build_vocab(episodes, mode="all_corpus")
    for cid, idx in vocab["id_to_index"].items():
        assert vocab["index_to_id"][idx] == cid
