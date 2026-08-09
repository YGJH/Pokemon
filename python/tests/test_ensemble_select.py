"""Greedy subset selection over ensemble members.

Two properties carry the whole feature and neither is visible at runtime:

* the cached-probability score must equal what a real ensemble eval would
  report, or ``--ensemble-top`` optimises a number nobody else computes;
* the ordering must be refused when it does not describe the checkpoints on
  disk, because the alternative — packaging the glob's first N — produces a
  submission that looks selected and is not.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ptcg_il.baselines import (
    BaselineMismatch,
    load_baselines,
    member_record,
    record_ensemble_baseline,
    record_ensemble_selection,
)
from ptcg_il.ensemble_select import (
    SelectionUnavailable,
    build_selection,
    greedy_order,
    select_from_baselines,
    subset_accuracy,
)


def _probs(rows: list[list[list[float]]]) -> np.ndarray:
    """[M, N, O] float32 from nested lists."""
    return np.asarray(rows, dtype=np.float32)


class TestGreedyOrder:
    def test_picks_the_complementary_member_over_the_better_one(self):
        # All three members score 1/2 alone.  Members 1 and 2 are identical, so
        # pairing them changes nothing; member 0 is wrong exactly where 1 is
        # right and *weakly* so, which is what lets the average flip both rows.
        # Any tie-broken top-2 could take {1, 2} and stay at 1/2; greedy must
        # reach 2/2 by pairing complements.
        targets = np.array([0, 1], dtype=np.int64)
        weak_wrong_strong_right = [[0.45, 0.55], [0.10, 0.90]]
        strong_right_weak_wrong = [[0.90, 0.10], [0.55, 0.45]]
        probs = _probs([
            weak_wrong_strong_right,   # member 0: wrong, right
            strong_right_weak_wrong,   # member 1: right, wrong
            strong_right_weak_wrong,   # member 2: a clone of member 1
        ])
        assert subset_accuracy(probs, targets, [0]) == 0.5
        assert subset_accuracy(probs, targets, [1]) == 0.5
        assert subset_accuracy(probs, targets, [1, 2]) == 0.5
        assert subset_accuracy(probs, targets, [0, 1]) == 1.0

        order, scores = greedy_order(probs, targets)
        assert len(order) == 3 and sorted(order) == [0, 1, 2]
        assert set(order[:2]) == {0, 1}, "greedy took the individually-best pair"
        assert scores[1] == 1.0

    def test_scores_track_the_prefix_of_the_order(self):
        rng = np.random.default_rng(0)
        probs = rng.random((5, 40, 6), dtype=np.float32)
        probs /= probs.sum(axis=-1, keepdims=True)
        targets = rng.integers(0, 6, size=40).astype(np.int64)

        order, scores = greedy_order(probs, targets)
        assert len(scores) == 5
        for k in range(1, 6):
            assert scores[k - 1] == pytest.approx(
                subset_accuracy(probs, targets, order[:k])
            )

    def test_monotone_nonincreasing_choice_is_still_recorded(self):
        # Greedy is not guaranteed to improve at every step; the recorded curve
        # has to show that rather than hide it, since "where does it peak" is
        # the question --ensemble-top exists to answer.
        rng = np.random.default_rng(7)
        probs = rng.random((6, 60, 4), dtype=np.float32)
        probs /= probs.sum(axis=-1, keepdims=True)
        targets = rng.integers(0, 4, size=60).astype(np.int64)
        _order, scores = greedy_order(probs, targets)
        assert len(scores) == 6
        assert max(scores) >= scores[-1]

    def test_ties_go_to_the_lowest_index(self):
        targets = np.array([0, 0], dtype=np.int64)
        identical = [[0.8, 0.2], [0.8, 0.2]]
        probs = _probs([identical, identical, identical])
        order, _ = greedy_order(probs, targets)
        assert order == [0, 1, 2], "tie-break must be deterministic across runs"

    def test_rejects_empty_and_mismatched_input(self):
        with pytest.raises(ValueError):
            greedy_order(np.zeros((2, 0, 3), np.float32), np.zeros((0,), np.int64))
        with pytest.raises(ValueError):
            greedy_order(np.zeros((2, 4, 3), np.float32), np.zeros((5,), np.int64))


class TestCachedScoreMatchesEnsemble:
    """The cache is only valid if it reproduces EnsemblePolicy's arg-max."""

    def test_matches_log_mean_prob_combination(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(3)
        n_members, n_rows, n_opt = 4, 50, 8

        logits = rng.normal(size=(n_members, n_rows, n_opt)).astype(np.float32) * 3.0
        opt_mask = rng.random((n_rows, n_opt)) > 0.3
        opt_mask[:, 0] = True  # keep at least one legal option per row
        # Non-trivial rows only, matching the filter collect_member_probs uses.
        opt_mask[opt_mask.sum(axis=-1) < 2, 1] = True

        t_mask = torch.from_numpy(opt_mask)
        cached = []
        for m in range(n_members):
            lg = torch.from_numpy(logits[m]).masked_fill(~t_mask, -1e9)
            cached.append(torch.softmax(lg, dim=-1).numpy())
        cached = np.stack(cached, axis=0)

        for subset in ([0], [0, 2], [1, 2, 3], [0, 1, 2, 3]):
            # What EnsemblePolicy.forward does (ensemble.py:160-179).
            probs = torch.stack([
                torch.softmax(
                    torch.from_numpy(logits[i]).masked_fill(~t_mask, -1e9), dim=-1
                )
                for i in subset
            ], dim=0).mean(dim=0)
            ens_logits = torch.log(probs + 1e-10).masked_fill(~t_mask, -1e9)
            ens_pred = ens_logits.argmax(dim=-1).numpy()

            cached_pred = cached[subset].mean(axis=0).argmax(axis=-1)
            assert (ens_pred == cached_pred).all(), (
                f"cached arg-max diverged from the ensemble for subset {subset}"
            )


@pytest.fixture
def ensemble_dir(tmp_path):
    """Four fake member checkpoints plus the data dir holding the baselines."""
    paths = []
    for i in range(4):
        p = tmp_path / f"ckpt-s{i}.pt"
        p.write_bytes(f"weights-{i}".encode())
        paths.append(str(p))
    return tmp_path, paths


class TestSelectFromBaselines:
    def _record(self, data_dir, paths, order, scores=None):
        members = [
            member_record(p, {"val/top1_nontrivial": 0.5 + 0.01 * i})
            for i, p in enumerate(paths)
        ]
        selection = build_selection(
            order, scores or [0.60, 0.71, 0.70, 0.69][:len(order)],
            paths, split="val", n_rows=1234,
        )
        return record_ensemble_selection(data_dir, 1, paths, members, selection)

    def test_returns_the_recorded_order_prefix(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        self._record(data_dir, paths, [2, 0, 3, 1])

        assert select_from_baselines(paths, 2, data_dir) == [paths[2], paths[0]]
        assert select_from_baselines(paths, 3, data_dir) == [
            paths[2], paths[0], paths[3]
        ]

    def test_order_is_independent_of_the_glob_order(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        self._record(data_dir, paths, [3, 1, 0, 2])
        shuffled = [paths[1], paths[3], paths[0], paths[2]]
        assert select_from_baselines(shuffled, 2, data_dir) == [paths[3], paths[1]]

    def test_refuses_when_no_selection_covers_the_member_set(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        self._record(data_dir, paths[:3], [1, 0, 2])
        with pytest.raises(SelectionUnavailable):
            select_from_baselines(paths, 2, data_dir)

    def test_does_not_match_a_record_holding_each_member_twice(self, ensemble_dir):
        # A glob expanded twice records 2N member_paths over N distinct files.
        # Matching that on set equality would slice an order indexing into the
        # 2N list and return duplicate checkpoints as "the best K".
        data_dir, paths = ensemble_dir
        doubled = paths + paths
        members = [member_record(p, {"val/top1_nontrivial": 0.5}) for p in doubled]
        record_ensemble_selection(
            data_dir, 1, doubled, members,
            build_selection(list(range(8)), [0.5] * 8, doubled,
                            split="val", n_rows=10),
        )
        with pytest.raises(SelectionUnavailable):
            select_from_baselines(paths, 2, data_dir)

    def test_refuses_a_bare_ensemble_record_with_no_selection(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        record_ensemble_baseline(data_dir, 1, paths, {"val/top1_nontrivial": 0.7})
        with pytest.raises(SelectionUnavailable):
            select_from_baselines(paths, 2, data_dir)

    def test_refuses_when_a_checkpoint_changed_since_selection(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        self._record(data_dir, paths, [0, 1, 2, 3])
        # Retrain one member in place — the ordering now describes weights that
        # no longer exist, and its prefix is not the best subset of what does.
        (data_dir / "ckpt-s1.pt").write_bytes(b"retrained-weights")
        with pytest.raises(BaselineMismatch):
            select_from_baselines(paths, 2, data_dir)
        # Unchanged members still resolve when the changed one is not in the set.
        assert select_from_baselines(
            paths, 2, data_dir, verify_sha1=False
        ) == [paths[0], paths[1]]

    def test_rejects_k_out_of_range(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        self._record(data_dir, paths, [0, 1, 2, 3])
        with pytest.raises(ValueError):
            select_from_baselines(paths, 0, data_dir)
        with pytest.raises(ValueError):
            select_from_baselines(paths, 9, data_dir)


class TestRecordMerging:
    def test_selection_survives_a_metrics_re_record(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        members = [member_record(p, {"val/top1_nontrivial": 0.5}) for p in paths]
        record_ensemble_selection(
            data_dir, 1, paths, members,
            build_selection([1, 0, 2, 3], [0.6, 0.7, 0.7, 0.7], paths,
                            split="val", n_rows=10),
        )
        record_ensemble_baseline(
            data_dir, 1, paths, {"val/top1_nontrivial": 0.72}, members=members,
        )
        rec = load_baselines(data_dir)["ens-4-1"]
        assert rec["selection"]["order"] == [1, 0, 2, 3]
        assert rec["nontrivial_top1"] == pytest.approx(0.72)

    def test_selection_is_dropped_when_the_member_set_changes(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        members = [member_record(p, {"val/top1_nontrivial": 0.5}) for p in paths]
        record_ensemble_selection(
            data_dir, 1, paths, members,
            build_selection([1, 0, 2, 3], [0.6, 0.7, 0.7, 0.7], paths,
                            split="val", n_rows=10),
        )
        other = list(paths)
        other[3] = str(data_dir / "ckpt-other.pt")
        (data_dir / "ckpt-other.pt").write_bytes(b"different-member")
        record_ensemble_baseline(data_dir, 1, other, {"val/top1_nontrivial": 0.4})

        rec = load_baselines(data_dir)["ens-4-1"]
        assert "selection" not in rec, (
            "an order indexing into the old member list would point at the "
            "wrong checkpoints"
        )

    def test_selection_records_the_split_it_was_fit_on(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        members = [member_record(p, {"val/top1_nontrivial": 0.5}) for p in paths]
        record_ensemble_selection(
            data_dir, 1, paths, members,
            build_selection([0, 1, 2, 3], [0.6, 0.65, 0.66, 0.64], paths,
                            split="val", n_rows=99),
        )
        rec = json.loads((data_dir / "il_baselines.json").read_text())["ens-4-1"]
        assert rec["selection"]["split"] == "val"
        assert rec["selection"]["metric"] == "nontrivial_top1"
        assert rec["selection"]["best_k"] == 3
        assert [m["path"] for m in rec["members"]] == [
            str(p) for p in rec["member_paths"]
        ]

    def test_member_table_length_must_match_the_paths(self, ensemble_dir):
        data_dir, paths = ensemble_dir
        members = [member_record(p, {"val/top1_nontrivial": 0.5}) for p in paths[:2]]
        with pytest.raises(ValueError):
            record_ensemble_baseline(
                data_dir, 1, paths, {"val/top1_nontrivial": 0.7}, members=members,
            )
