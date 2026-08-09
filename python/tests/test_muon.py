"""Vendored Muon: Newton-Schulz orthogonalization, and the two-group optimizer.

Muon replaces the momentum update for 2D hidden weights with its *orthogonalized*
version -- Newton-Schulz drives every singular value toward 1, so the update's
scale stops depending on the gradient's scale and starts depending only on its
direction.  Everything Muon cannot sensibly orthogonalize (embeddings, biases,
norms, rank-deficient heads) stays on AdamW in the same optimizer.

Three properties carry the whole thing, and each fails silently if broken:

*Orthogonalization.*  Wrong Newton-Schulz coefficients still return a matrix of
the right shape and still train -- just badly, as an oddly-conditioned SGD.  The
test is on the singular values.

*Descent direction.*  The orthogonalized update must still point downhill.  A
sign error or a stray transpose gives a well-conditioned matrix that ascends.

*Faithful AdamW.*  The non-Muon half is hand-rolled so both algorithms can live
in one optimizer and one scheduler.  It is asserted against ``torch.optim.AdamW``
step for step, because a subtly different bias correction would show up only as
a slightly worse baseline -- and the whole point of this change is an A/B
against that baseline.
"""

import math

import pytest
import torch
import torch.nn as nn

from ptcg_il.train.muon import Muon, zeropower_via_newtonschulz5


def _svals(m: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(m.float())


class TestNewtonSchulz:
    @pytest.mark.parametrize("shape", [(64, 64), (32, 128), (128, 32), (256, 7)])
    def test_flattens_the_spectrum_and_bounds_it(self, shape):
        """The defining property: the output is (approximately) orthogonal.

        Five quintic iterations do not converge exactly, and the tuned
        coefficients are not trying to -- they flatten the spectrum as fast as
        possible and leave it in a band near 1.  A square Gaussian is very
        nearly singular (condition ~3000 here) and five steps take it to ~19,
        not to 1; a well-conditioned input lands in [0.68, 1.06].  So the
        falsifiable claims are: the spectrum is bounded above, and it is
        dramatically less spread than it started.  Wrong coefficients either
        diverge (max grows) or barely move the conditioning.
        """
        torch.manual_seed(0)
        g = torch.randn(*shape)
        before = _svals(g)
        cond_before = (before.max() / before.min()).item()

        s = _svals(zeropower_via_newtonschulz5(g, steps=5))
        cond_after = (s.max() / s.min()).item()

        assert s.max() < 1.3, f"singular values blew up: max={s.max():.3f}"
        if cond_before > 100:
            assert cond_after < cond_before / 10, (
                f"spectrum barely flattened: {cond_before:.0f} -> {cond_after:.1f}"
            )
        else:
            assert cond_after < 2.0, (
                f"spectrum not flattened: {cond_before:.2f} -> {cond_after:.2f}"
            )
            assert s.min() > 0.5, f"singular values collapsed: min={s.min():.3f}"

    def test_stays_a_descent_direction(self):
        """<G, ortho(G)> > 0 -- a transpose or sign slip silently ascends."""
        torch.manual_seed(1)
        n_checked = 0
        for shape in [(64, 64), (16, 96), (96, 16)]:
            g = torch.randn(*shape)
            o = zeropower_via_newtonschulz5(g, steps=5).float()
            assert (g * o).sum() > 0, f"{shape} update is not downhill"
            n_checked += 1
        assert n_checked == 3

    def test_is_scale_invariant(self):
        """Scaling the gradient must not scale the update.

        This is the property the whole optimizer is built on: it is what makes
        the update size a function of direction alone, and what makes grad
        clipping redundant for these parameters.
        """
        torch.manual_seed(2)
        g = torch.randn(64, 64)
        small = zeropower_via_newtonschulz5(g * 1e-3, steps=5).float()
        large = zeropower_via_newtonschulz5(g * 1e3, steps=5).float()
        assert torch.allclose(small, large, atol=2e-2)

    def test_preserves_shape_for_wide_and_tall(self):
        for shape in [(8, 128), (128, 8)]:
            assert zeropower_via_newtonschulz5(torch.randn(*shape)).shape == shape

    def test_runs_in_bf16_without_nan(self):
        """The iteration runs in bf16 by design; it must not overflow."""
        torch.manual_seed(3)
        out = zeropower_via_newtonschulz5(torch.randn(128, 128) * 1e4, steps=5)
        assert torch.isfinite(out.float()).all()

    def test_rejects_a_1d_tensor(self):
        """A rank-1 parameter has no spectrum to flatten; caller must partition."""
        with pytest.raises((AssertionError, ValueError, RuntimeError)):
            zeropower_via_newtonschulz5(torch.randn(64))


class TestMuonGroup:
    def _one_param(self, shape=(32, 32), seed=0):
        torch.manual_seed(seed)
        p = nn.Parameter(torch.randn(*shape))
        return p

    def test_update_magnitude_is_insensitive_to_gradient_scale(self):
        """SGD/Adam would move 1000x further for a 1000x gradient; Muon does not."""
        deltas = []
        for scale in (1e-3, 1e3):
            p = self._one_param()
            start = p.detach().clone()
            opt = Muon([{"params": [p], "use_muon": True, "lr": 0.02,
                         "weight_decay": 0.0}])
            p.grad = torch.ones_like(p) * scale + torch.randn_like(p) * scale
            opt.step()
            deltas.append((p.detach() - start).norm().item())
        ratio = max(deltas) / min(deltas)
        assert ratio < 1.5, f"update scaled with the gradient ({ratio:.1f}x)"

    def test_actually_descends_a_quadratic(self):
        torch.manual_seed(4)
        target = torch.randn(32, 32)
        p = nn.Parameter(torch.zeros(32, 32))
        opt = Muon([{"params": [p], "use_muon": True, "lr": 0.05,
                     "weight_decay": 0.0}])
        # Orthogonal updates have a fixed per-step size, so convergence from
        # zeros to a random target is steady rather than fast: measured
        # 1.050 -> 0.107 over 120 steps at lr=0.05.
        losses = []
        for _ in range(120):
            loss = ((p - target) ** 2).mean()
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert losses[-1] < losses[0] * 0.2, f"{losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_nesterov_flag_changes_the_update(self):
        """Both settings descend, so only a direct comparison catches a no-op.

        Nesterov looks ahead by blending the fresh gradient into the buffer
        before orthogonalizing; ignoring the flag and always using the raw
        buffer trains perfectly well and just is not the algorithm.
        """
        outs = []
        for nesterov in (True, False):
            p = self._one_param(seed=11)
            opt = Muon([{"params": [p], "use_muon": True, "lr": 0.02,
                         "weight_decay": 0.0, "nesterov": nesterov}])
            torch.manual_seed(12)
            for _ in range(3):
                p.grad = torch.randn_like(p)
                opt.step()
            outs.append(p.detach().clone())
        assert not torch.allclose(outs[0], outs[1], atol=1e-6), (
            "nesterov=True and nesterov=False produced the same weights"
        )

    def test_momentum_state_is_kept_per_parameter(self):
        p = self._one_param()
        opt = Muon([{"params": [p], "use_muon": True, "lr": 0.02,
                     "weight_decay": 0.0}])
        p.grad = torch.randn_like(p)
        opt.step()
        assert "momentum_buffer" in opt.state[p]

    def test_weight_decay_shrinks_a_zero_gradient_parameter(self):
        p = self._one_param()
        start = p.detach().clone()
        opt = Muon([{"params": [p], "use_muon": True, "lr": 0.1,
                     "weight_decay": 0.5}])
        p.grad = torch.zeros_like(p)
        opt.step()
        assert p.detach().abs().sum() < start.abs().sum()


class TestAdamWGroupMatchesTorch:
    """The non-Muon half must be torch's AdamW, not something like it.

    It is hand-rolled only so that both algorithms share one optimizer object
    and therefore one LambdaLR.  Any divergence would move the baseline this
    change is supposed to be measured against.
    """

    def _run(self, opt_factory, steps=6, seed=7):
        torch.manual_seed(seed)
        p = nn.Parameter(torch.randn(16, 24))
        opt = opt_factory([p])
        torch.manual_seed(99)
        grads = [torch.randn(16, 24) for _ in range(steps)]
        for g in grads:
            p.grad = g.clone()
            opt.step()
        return p.detach()

    def test_matches_torch_adamw_step_for_step(self):
        mine = self._run(lambda ps: Muon(
            [{"params": ps, "use_muon": False, "lr": 3e-4,
              "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-8}]))
        torch_ref = self._run(lambda ps: torch.optim.AdamW(
            ps, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8))
        assert torch.allclose(mine, torch_ref, atol=1e-7, rtol=0), (
            f"max |Δ| = {(mine - torch_ref).abs().max():.3e}"
        )

    def test_matches_torch_adamw_with_zero_weight_decay(self):
        mine = self._run(lambda ps: Muon(
            [{"params": ps, "use_muon": False, "lr": 1e-3,
              "weight_decay": 0.0, "betas": (0.9, 0.999), "eps": 1e-8}]))
        torch_ref = self._run(lambda ps: torch.optim.AdamW(
            ps, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8))
        assert torch.allclose(mine, torch_ref, atol=1e-7, rtol=0)


class TestMixedOptimizer:
    def _model(self):
        torch.manual_seed(5)
        return nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 4))

    def test_each_group_uses_its_own_algorithm(self):
        """A matrix and a bias in one optimizer must not share an update rule."""
        m = self._model()
        mats = [m[0].weight, m[2].weight]
        rest = [m[0].bias, m[2].bias, m[1].weight, m[1].bias]
        opt = Muon([
            {"params": mats, "use_muon": True, "lr": 0.02, "weight_decay": 0.0},
            {"params": rest, "use_muon": False, "lr": 3e-4, "weight_decay": 0.0},
        ])
        for p in mats + rest:
            p.grad = torch.randn_like(p)
        opt.step()
        assert "momentum_buffer" in opt.state[mats[0]]
        assert "exp_avg_sq" in opt.state[rest[0]]
        assert "exp_avg_sq" not in opt.state[mats[0]]

    def test_lambdalr_scales_both_groups(self):
        """One scheduler drives both base LRs — that is why they share an object."""
        m = self._model()
        opt = Muon([
            {"params": [m[0].weight], "use_muon": True, "lr": 0.02,
             "weight_decay": 0.0},
            {"params": [m[0].bias], "use_muon": False, "lr": 3e-4,
             "weight_decay": 0.0},
        ])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)
        sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
        assert opt.param_groups[1]["lr"] == pytest.approx(1.5e-4)

    def test_state_dict_roundtrips(self):
        m = self._model()
        def build():
            return Muon([
                {"params": [m[0].weight], "use_muon": True, "lr": 0.02,
                 "weight_decay": 0.0},
                {"params": [m[0].bias], "use_muon": False, "lr": 3e-4,
                 "weight_decay": 0.0},
            ])
        opt = build()
        m[0].weight.grad = torch.randn_like(m[0].weight)
        m[0].bias.grad = torch.randn_like(m[0].bias)
        opt.step()
        sd = opt.state_dict()
        opt2 = build()
        opt2.load_state_dict(sd)
        assert opt2.param_groups[0]["use_muon"] is True
        assert "momentum_buffer" in opt2.state[m[0].weight]

    def test_a_parameter_without_a_gradient_is_skipped(self):
        m = self._model()
        opt = Muon([{"params": [m[0].weight, m[2].weight], "use_muon": True,
                     "lr": 0.02, "weight_decay": 0.0}])
        before = m[2].weight.detach().clone()
        m[0].weight.grad = torch.randn_like(m[0].weight)
        opt.step()
        assert torch.equal(m[2].weight.detach(), before)


class TestStackedMatrices:
    """Fused QKV and GRU gates are several maps concatenated on the output axis.

    ``[768, 256]`` is not a 768->256 map; it is three 256->256 maps stacked.
    Orthogonalizing the stack flattens a *joint* spectrum — the stack has rank
    <= 256, so the three blocks end up sharing one normalization and stop being
    independently conditioned.  On this model that would be 28.6% of parameters
    (9 attention projections plus 4 GRU gate stacks) getting an algorithm nobody
    intended.
    """

    def _blocky_grad(self, rows=256, cols=256, n=3, seed=0):
        """A gradient whose blocks differ in scale by 1000x."""
        torch.manual_seed(seed)
        return torch.cat(
            [torch.randn(rows, cols) * s for s in (1e3, 1.0, 1e-3)][:n], dim=0
        )

    def test_split_normalizes_each_block_independently(self):
        p = nn.Parameter(torch.zeros(768, 256))
        opt = Muon([{"params": [p], "use_muon": True, "lr": 0.02,
                     "weight_decay": 0.0, "split": 3}])
        p.grad = self._blocky_grad()
        opt.step()
        norms = [p.detach()[i * 256:(i + 1) * 256].norm().item() for i in range(3)]
        assert max(norms) / min(norms) < 1.2, (
            f"blocks were not independently normalized: {norms}"
        )

    def test_without_split_the_loud_block_dominates(self):
        """The behaviour the split exists to avoid — pinned so the test is not vacuous."""
        p = nn.Parameter(torch.zeros(768, 256))
        opt = Muon([{"params": [p], "use_muon": True, "lr": 0.02,
                     "weight_decay": 0.0, "split": 1}])
        p.grad = self._blocky_grad()
        opt.step()
        norms = [p.detach()[i * 256:(i + 1) * 256].norm().item() for i in range(3)]
        assert max(norms) / min(norms) > 5, (
            f"joint normalization should have skewed the blocks: {norms}"
        )

    def test_split_update_matches_the_equivalent_separate_parameters(self):
        """A split [768,256] must equal three [256,256] params stepped alone.

        This is the whole claim, and it also pins the shape-scale factor: the
        aspect-ratio correction has to be computed per *slice*, or the stacked
        version picks up a spurious sqrt(3).
        """
        torch.manual_seed(1)
        g = torch.randn(768, 256)

        stacked = nn.Parameter(torch.zeros(768, 256))
        opt_s = Muon([{"params": [stacked], "use_muon": True, "lr": 0.02,
                       "weight_decay": 0.0, "split": 3}])
        stacked.grad = g.clone()
        opt_s.step()

        separate = [nn.Parameter(torch.zeros(256, 256)) for _ in range(3)]
        opt_p = Muon([{"params": separate, "use_muon": True, "lr": 0.02,
                       "weight_decay": 0.0, "split": 1}])
        for i, q in enumerate(separate):
            q.grad = g[i * 256:(i + 1) * 256].clone()
        opt_p.step()

        want = torch.cat([q.detach() for q in separate], dim=0)
        assert torch.allclose(stacked.detach(), want, atol=1e-6), (
            f"max |Δ| = {(stacked.detach() - want).abs().max():.3e}"
        )

    def test_split_must_divide_the_output_dimension(self):
        p = nn.Parameter(torch.zeros(100, 32))
        opt = Muon([{"params": [p], "use_muon": True, "lr": 0.02,
                     "weight_decay": 0.0, "split": 3}])
        p.grad = torch.randn_like(p)
        with pytest.raises(ValueError, match="divide"):
            opt.step()

    def test_split_defaults_to_one(self):
        p = nn.Parameter(torch.zeros(64, 64))
        opt = Muon([{"params": [p], "use_muon": True, "lr": 0.02,
                     "weight_decay": 0.0}])
        p.grad = torch.randn_like(p)
        opt.step()
        assert torch.isfinite(p.detach()).all()
