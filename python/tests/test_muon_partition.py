"""Which parameters Muon may touch, and the guarantee that none is touched twice.

Muon orthogonalizes a matrix's update, which is meaningful for a linear map in
the trunk and meaningless — or actively wrong — for three other things this
model contains:

* ``nn.Embedding`` tables, where rows are independent lookups, not a map;
* 1D biases and norm gains, which have no spectrum;
* **output heads**, whose weight is ``[few, D]``.  ``pointer.score`` is
  ``[1, 256]``: Newton-Schulz on a rank-1 map does nothing but set its norm to
  1, which silently rescales every pointer logit.  This model has already had
  one logit-scale blow-up that grad clipping could not stop, so that is not a
  side effect to acquire by accident.

The partition is by **out-features**, not by ``min(shape)``, and that distinction
is the point of these tests: ``embed.hand_mlp.0.weight`` is ``[256, 7]`` and
belongs on Muon (a real 7->256 map, which is exactly what the aspect-ratio scale
factor exists for), while ``belief_heads.arch_head.2.weight`` is ``[9, 256]`` and
does not.  A ``min(shape)`` rule cannot tell those apart.
"""

import pytest
import torch
import torch.nn as nn

from ptcg_il.train.loop import create_optimizer, partition_parameters
from ptcg_il.train.muon import Muon
from tests.test_model_policy import make_policy


@pytest.fixture(scope="module")
def policy():
    return make_policy(D=64, heads=4, layers=2, ff=128, n_opp_arch=9)


def _named(policy, params):
    ids = {id(p) for p in params}
    return {n for n, p in policy.named_parameters() if id(p) in ids}


class TestPartition:
    def test_every_parameter_lands_in_exactly_one_group(self, policy):
        """Two groups holding one tensor would update it twice per step.

        Not hypothetical here: one ``CardEncoder`` is aliased into
        ``pointer.card``, ``belief.card_emb`` and ``belief_heads``, so the same
        tensors are reachable under several names.
        """
        groups = partition_parameters(policy)
        seen: dict[int, str] = {}
        for kind, params in groups.items():
            for p in params:
                assert id(p) not in seen, (
                    f"parameter in both {seen[id(p)]} and {kind}"
                )
                seen[id(p)] = kind

        all_params = {id(p) for p in policy.parameters()}
        assert set(seen) == all_params, (
            f"{len(all_params - set(seen))} parameter(s) in no group"
        )

    def test_the_trunk_matrices_go_to_muon(self, policy):
        """The encoder is the bulk of the model; it must all be on Muon."""
        muon = _named(policy, partition_parameters(policy)["muon"])
        assert muon, "nothing was assigned to Muon"
        for n in muon:
            # belief_heads.arch_head.0 is [D, D] -- the head's *hidden* layer,
            # a real map.  Only its [9, D] output layer is carved out.
            assert n.startswith(
                ("encoder.", "embed.", "pointer.", "belief.", "belief_heads.",
                 "value.")
            ), n

        encoder_mats = {
            n for n, p in policy.named_parameters()
            if n.startswith("encoder.") and p.ndim == 2
        }
        assert encoder_mats, "fixture has no encoder matrices"
        assert encoder_mats <= muon, (
            f"encoder matrices left off Muon: {sorted(encoder_mats - muon)}"
        )

    def test_output_heads_and_embeddings_stay_on_adamw(self, policy):
        groups = partition_parameters(policy)
        adamw = _named(policy, groups["adamw_decay"]) | _named(
            policy, groups["adamw_no_decay"])
        muon = _named(policy, groups["muon"])

        n_checked = 0
        for name in ("pointer.score.weight", "value.f.2.weight",
                     "belief_heads.arch_head.2.weight", "belief_heads.card_bias"):
            assert name in adamw, f"{name} must not be orthogonalized"
            assert name not in muon
            n_checked += 1
        assert n_checked == 4

        for m_name, m in policy.named_modules():
            if isinstance(m, nn.Embedding):
                assert f"{m_name}.weight" in adamw, f"{m_name} must stay on AdamW"
                n_checked += 1
        assert n_checked > 4, "no embeddings examined"

    def test_a_wide_input_projection_is_muon_not_a_head(self, policy):
        """[256, 7] is a trunk map; [9, 256] is a head.  min(shape) cannot tell."""
        muon = _named(policy, partition_parameters(policy)["muon"])
        wide = [n for n, p in policy.named_parameters()
                if p.ndim == 2 and p.size(-2) >= 64 and p.size(-1) <= 16]
        assert wide, "fixture has no narrow-fan-in projection to test"
        for n in wide:
            assert n in muon, f"{n} is a trunk projection and belongs on Muon"

    def test_no_muon_parameter_is_rank_starved(self, policy):
        """The guard that keeps the rule from going stale.

        A head added later would otherwise be orthogonalized silently.
        """
        for p in partition_parameters(policy)["muon"]:
            assert p.ndim == 2
            assert p.size(-2) >= 16, f"out-features {p.size(-2)} is a head, not a map"

    def test_weight_decay_split_still_honours_no_decay(self, policy):
        groups = partition_parameters(policy)
        no_decay = _named(policy, groups["adamw_no_decay"])
        assert no_decay
        for n in no_decay:
            assert any(t in n for t in ("bias", "norm", "emb", "ln_", "null", "no_stadium")), n


class TestCreateOptimizer:
    def test_muon_mode_builds_both_group_kinds(self, policy):
        opt = create_optimizer(policy, optimizer="muon", peak_lr=3e-4, muon_lr=0.02)
        assert isinstance(opt, Muon)
        kinds = {g["use_muon"] for g in opt.param_groups}
        assert kinds == {True, False}
        for g in opt.param_groups:
            assert g["lr"] == (0.02 if g["use_muon"] else 3e-4)

    def test_adamw_mode_is_unchanged(self, policy):
        """The A/B baseline must still be the optimizer it always was."""
        opt = create_optimizer(policy, optimizer="adamw", peak_lr=3e-4)
        assert isinstance(opt, torch.optim.AdamW)
        assert len(opt.param_groups) == 2

    def test_default_is_still_adamw(self, policy):
        assert isinstance(create_optimizer(policy), torch.optim.AdamW)

    def test_unknown_optimizer_raises(self, policy):
        with pytest.raises(ValueError, match="optimizer"):
            create_optimizer(policy, optimizer="lion")

    def test_a_muon_step_changes_the_trunk_and_the_heads(self, policy):
        """End to end: both branches actually move their own parameters."""
        opt = create_optimizer(policy, optimizer="muon", peak_lr=3e-4, muon_lr=0.02)
        trunk = dict(policy.named_parameters())["encoder.enc.layers.0.linear1.weight"]
        head = dict(policy.named_parameters())["pointer.score.weight"]
        t0, h0 = trunk.detach().clone(), head.detach().clone()
        for p in policy.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
        assert not torch.equal(trunk.detach(), t0), "trunk did not move"
        assert not torch.equal(head.detach(), h0), "head did not move"
        assert "momentum_buffer" in opt.state[trunk]
        assert "exp_avg_sq" in opt.state[head]


class TestStackedDetection:
    """Fused QKV and GRU gates must be found automatically, by module type.

    28.6% of this model is stacked: 9 ``in_proj_weight`` at ``[3D, D]`` and 4
    GRU gate matrices at ``[3H, *]``.  Detecting them by shape would be
    guesswork — ``[768, 256]`` is only "three stacked maps" because of what
    module owns it.
    """

    def test_attention_and_gru_weights_are_marked_three_way(self, policy):
        from ptcg_il.train.loop import stacked_slice_counts

        slices = stacked_slice_counts(policy)
        by_name = {n: slices.get(id(p), 1) for n, p in policy.named_parameters()}

        qkv = [n for n in by_name if "in_proj_weight" in n]
        gru = [n for n in by_name if "gru" in n.lower() and "weight" in n]
        assert qkv, "fixture has no fused attention projection"
        assert gru, "fixture has no recurrent gate matrix"
        for n in qkv + gru:
            assert by_name[n] == 3, f"{n} should be split 3 ways, got {by_name[n]}"

    def test_ordinary_linears_are_not_split(self, policy):
        from ptcg_il.train.loop import stacked_slice_counts

        slices = stacked_slice_counts(policy)
        plain = dict(policy.named_parameters())["encoder.enc.layers.0.linear1.weight"]
        assert slices.get(id(plain), 1) == 1

    def test_every_split_count_divides_its_parameter(self, policy):
        from ptcg_il.train.loop import stacked_slice_counts

        n_checked = 0
        for p, n in ((p, n) for p, n in stacked_slice_counts(policy).items()):
            pass
        by_id = {id(p): p for p in policy.parameters()}
        for pid, n in stacked_slice_counts(policy).items():
            assert by_id[pid].size(-2) % n == 0
            n_checked += 1
        assert n_checked > 0

    def test_create_optimizer_gives_stacked_params_their_own_group(self, policy):
        opt = create_optimizer(policy, optimizer="muon", peak_lr=3e-4, muon_lr=0.02)
        muon_groups = [g for g in opt.param_groups if g["use_muon"]]
        splits = {g.get("split", 1) for g in muon_groups}
        assert splits == {1, 3}, f"expected split-1 and split-3 groups, got {splits}"
        for g in muon_groups:
            assert g["lr"] == 0.02, "both Muon groups share one learning rate"

    def test_all_muon_params_still_appear_exactly_once(self, policy):
        opt = create_optimizer(policy, optimizer="muon", peak_lr=3e-4, muon_lr=0.02)
        seen = []
        for g in opt.param_groups:
            seen.extend(id(p) for p in g["params"])
        assert len(seen) == len(set(seen)), "a parameter is in two groups"
        assert set(seen) == {id(p) for p in policy.parameters()}


class TestResumeAcrossOptimizerChange:
    """Switching optimizer invalidates the optimizer state, and only that state.

    AdamW keeps two moments per parameter; Muon keeps one momentum buffer for
    the trunk and two moments for the rest.  ``load_state_dict`` matches groups
    by position and count, so feeding one to the other either raises deep inside
    torch with a shape message that names nothing useful, or — when the group
    counts happen to line up — loads moments into a momentum buffer and carries
    on.  The checkpoint therefore records which optimizer wrote it, and the
    resume path refuses a mismatch by name.
    """

    def _ckpt(self, tmp_path, policy, optimizer, tag):
        from ptcg_il.train.checkpoint import save_checkpoint
        from ptcg_il.train.loop import create_schedule

        class _EMAStub:
            def state_dict(self):
                return {}

        opt = create_optimizer(policy, optimizer=optimizer)
        sched = create_schedule(opt, total_steps=10)
        return save_checkpoint(
            policy, opt, sched, _EMAStub(), step=1, save_dir=tmp_path,
            tag=tag,
            # save_checkpoint refuses a deck record without the 60 card ids.
            deck={"archetype_self": 0, "deck": list(range(1, 61)),
                  "vocab_sha1": "x", "archetypes_sha1": "y"},
        )

    def test_checkpoint_records_which_optimizer_wrote_it(self, tmp_path, policy):
        import torch as _t

        for kind in ("adamw", "muon"):
            path = self._ckpt(tmp_path, policy, kind, kind)
            ckpt = _t.load(path, map_location="cpu", weights_only=False)
            assert ckpt["optimizer"] == kind

    def test_resuming_adamw_state_into_muon_raises(self, tmp_path, policy):
        import torch as _t
        from ptcg_il.train.loop import require_matching_optimizer

        ckpt = _t.load(self._ckpt(tmp_path, policy, "adamw", "a"),
                       map_location="cpu", weights_only=False)
        with pytest.raises(ValueError, match="--optimizer"):
            require_matching_optimizer(ckpt, "muon")

    def test_resuming_muon_state_into_adamw_raises(self, tmp_path, policy):
        import torch as _t
        from ptcg_il.train.loop import require_matching_optimizer

        ckpt = _t.load(self._ckpt(tmp_path, policy, "muon", "m"),
                       map_location="cpu", weights_only=False)
        with pytest.raises(ValueError, match="--optimizer"):
            require_matching_optimizer(ckpt, "adamw")

    def test_matching_optimizer_is_accepted(self, tmp_path, policy):
        import torch as _t
        from ptcg_il.train.loop import require_matching_optimizer

        for kind in ("adamw", "muon"):
            ckpt = _t.load(self._ckpt(tmp_path, policy, kind, f"ok-{kind}"),
                           map_location="cpu", weights_only=False)
            require_matching_optimizer(ckpt, kind)  # must not raise

    def test_a_checkpoint_predating_the_record_is_assumed_adamw(self, policy):
        """Every checkpoint on disk today was written by AdamW."""
        from ptcg_il.train.loop import require_matching_optimizer

        require_matching_optimizer({}, "adamw")
        with pytest.raises(ValueError, match="--optimizer"):
            require_matching_optimizer({}, "muon")

    def test_muon_state_actually_roundtrips_into_a_muon_run(self, tmp_path, policy):
        """The refusal must not be hiding a resume path that never worked."""
        import torch as _t

        path = self._ckpt(tmp_path, policy, "muon", "rt")
        ckpt = _t.load(path, map_location="cpu", weights_only=False)
        fresh = create_optimizer(policy, optimizer="muon")
        fresh.load_state_dict(ckpt["optimizer_state_dict"])
        assert [g["use_muon"] for g in fresh.param_groups] == [True, True, False, False]
