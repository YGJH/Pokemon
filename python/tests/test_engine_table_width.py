"""A stale engine table must fail loudly, not produce an empty corpus.

`ptcg_mine.cards` writes `engine_card_features.npy` / `engine_attack_features.npy`
at whatever `F_CARD`/`F_ATK` were when mining last ran.  Editing the featurizer
changes those widths, so the tables on disk go stale until mining re-runs.

What that used to look like: `featurize` raises
``ValueError: could not broadcast input array from shape (218,) into shape (223,)``
on **every** decision, `shard_writer`'s pass-B worker catches ValueError and
`logger.debug`s it, and 30 minutes later the build reports
"0 total samples" with an error message about experts and D_self/D_opp -- none
of which had anything to do with it.  Measured for real: 125,123 kept pairs, 0
samples, no traceback.
"""
import numpy as np
import pytest


def _write_tables(tmp_path, card_width, attack_width):
    np.save(tmp_path / "engine_card_features.npy",
            {1: np.zeros(card_width, dtype=np.float32),
             2: np.zeros(card_width, dtype=np.float32)})
    np.save(tmp_path / "engine_attack_features.npy",
            {1: np.zeros(attack_width, dtype=np.float32)})


class TestEngineTableWidthValidation:
    def test_correct_widths_load_fine(self, tmp_path):
        from ptcg_il.featurizer import F_ATK, F_CARD, load_engine_tables

        _write_tables(tmp_path, F_CARD, F_ATK)
        t = load_engine_tables(tmp_path)
        assert t["engine_card_features"] is not None
        assert t["engine_attack_features"] is not None

    def test_stale_card_table_raises_with_the_remedy(self, tmp_path):
        from ptcg_il.featurizer import F_ATK, F_CARD, load_engine_tables

        _write_tables(tmp_path, F_CARD - 5, F_ATK)
        with pytest.raises(ValueError) as ei:
            load_engine_tables(tmp_path)
        msg = str(ei.value)
        assert "engine_card_features" in msg
        assert str(F_CARD) in msg and str(F_CARD - 5) in msg, (
            "the error must name both widths so the mismatch is readable"
        )
        assert "mine" in msg, "the error must name the command that fixes it"

    def test_stale_attack_table_raises(self, tmp_path):
        from ptcg_il.featurizer import F_ATK, F_CARD, load_engine_tables

        _write_tables(tmp_path, F_CARD, F_ATK - 1)
        with pytest.raises(ValueError) as ei:
            load_engine_tables(tmp_path)
        assert "engine_attack_features" in str(ei.value)

    def test_absent_tables_are_still_allowed(self, tmp_path):
        """None means 'no information' for the one caller that has none."""
        from ptcg_il.featurizer import load_engine_tables

        t = load_engine_tables(tmp_path)
        assert t["engine_card_features"] is None
        assert t["engine_attack_features"] is None

    def test_the_real_tables_on_disk_match_the_featurizer(self):
        """Guards the actual data dir, so a stale rebuild is caught by CI."""
        from pathlib import Path

        from ptcg_il.featurizer import load_engine_tables

        data = Path(__file__).resolve().parent.parent / "data"
        if not (data / "engine_card_features.npy").exists():
            pytest.skip("no built data dir")
        load_engine_tables(data)  # raises if stale
