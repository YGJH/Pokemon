"""Muon — orthogonalized-momentum optimizer, vendored.

Muon (Jordan et al., 2024) takes the ordinary momentum update for a 2D weight
and replaces it with its *orthogonalization*: every singular value is driven
toward 1, so the update carries the gradient's direction and none of its scale.
For a transformer trunk that is worth a meaningful reduction in steps-to-target
over AdamW.

Vendored rather than added as a dependency: the algorithm is ~40 lines, it has
to interoperate with this repo's param-group conventions (``_no_decay``), and a
pip package would be a new supply-chain edge for something this small.

**Why one optimizer and not two.**  Muon deliberately does not apply to
everything.  Embeddings, biases and norm gains are not matrices whose spectrum
means anything, and neither are rank-deficient heads.  Those stay on AdamW.
Rather than run two optimizer objects — which would need two schedulers, two
state dicts and a wrapper to fake the ``Optimizer`` API for ``LambdaLR`` — both
algorithms live here, selected per param group by ``use_muon``.  That keeps
``create_schedule``, ``save_checkpoint`` and the resume path working unchanged.

The AdamW branch is therefore hand-rolled, and is asserted against
``torch.optim.AdamW`` step for step in ``test_muon.py``.  That test is not
ceremony: this change is judged by an A/B against an AdamW baseline, and a
subtly different bias correction would move the thing being measured.
"""

from __future__ import annotations

import torch

#: Quintic Newton-Schulz coefficients from the reference implementation.  They
#: are *not* the coefficients that converge to the exact orthogonalization —
#: they are tuned to flatten the spectrum as fast as possible in five steps,
#: which leaves the singular values in a band around 1 rather than at 1.  That
#: is deliberate and sufficient: the update only needs to stop carrying the
#: gradient's scale, not to be exactly orthogonal.
_NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

#: Iterations of the quintic.  Five is the reference value; the cost is 5 pairs
#: of matmuls per matrix per step, which against an 8-layer D=256 trunk is
#: negligible next to the forward/backward.
NS_STEPS: int = 5


@torch.no_grad()
def zeropower_via_newtonschulz5(
    G: torch.Tensor, steps: int = NS_STEPS, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the orthogonal factor of *G* (``U @ V.T`` of its SVD).

    Runs in bfloat16 by design.  The iteration is a fixed polynomial in
    ``X @ X.T`` and is numerically forgiving; fp32 costs roughly double for no
    measurable difference in the resulting update.

    The initial division by the Frobenius norm is what makes the whole thing
    scale-free — it is also why the caller may hand this an arbitrarily scaled
    gradient without clipping first.

    Tall matrices are transposed before the iteration and back after, so the
    quintic always runs on the smaller Gram matrix.
    """
    if G.ndim < 2:
        raise ValueError(
            f"Newton-Schulz needs a matrix, got shape {tuple(G.shape)}. "
            "1D parameters (biases, norm gains) belong in the AdamW group — "
            "see ptcg_il.train.loop.partition_parameters."
        )
    a, b, c = _NS_COEFFS
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Orthogonalized momentum for matrices, AdamW for everything else.

    Every param group must carry ``use_muon``.  Muon groups additionally read
    ``momentum`` and ``nesterov``; AdamW groups read ``betas`` and ``eps``.
    Both read ``lr`` and ``weight_decay`` (decoupled in both branches).

    The Muon learning rate is **not** on the same scale as an AdamW learning
    rate and does not transfer from it.  Because the update is orthogonal, its
    per-element size is set by the shape of the matrix rather than by the loss
    surface, so the two groups are given independent base LRs and a single
    scheduler scales both.
    """

    def __init__(
        self,
        param_groups,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = NS_STEPS,
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            nesterov=nesterov, betas=betas, eps=eps, ns_steps=ns_steps,
            use_muon=True, split=1,
        )
        super().__init__(param_groups, defaults)
        for i, group in enumerate(self.param_groups):
            if "use_muon" not in group:
                raise ValueError(
                    f"param group {i} has no 'use_muon' flag. Every group must "
                    "say which algorithm it wants — defaulting would silently "
                    "orthogonalize biases or leave the trunk on AdamW."
                )

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    # ------------------------------------------------------------------
    def _step_muon(self, group) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        momentum, nesterov = group["momentum"], group["nesterov"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.lerp_(g, 1.0 - momentum)
            update = g.lerp(buf, momentum) if nesterov else buf

            n_split = group.get("split", 1)
            update = self._orthogonalize(update, n_split, group["ns_steps"])

            # Shape correction, computed per *slice*.  Orthogonalization fixes
            # the spectrum, so the per-element size of the update still depends
            # on the aspect ratio: a tall matrix spreads the same spectral norm
            # over more rows.  This is the reference `max(1, rows/cols) ** 0.5`
            # factor.  Using the stacked shape here instead of the slice shape
            # would put a spurious sqrt(3) on every fused QKV and GRU gate.
            rows = p.size(-2) // n_split
            scale = max(1.0, rows / p.size(-1)) ** 0.5
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(update.reshape(p.shape).to(p.dtype), alpha=-lr * scale)

    @staticmethod
    def _orthogonalize(update: torch.Tensor, n_split: int, ns_steps: int):
        """Orthogonalize *update*, treating it as *n_split* stacked maps.

        A fused QKV projection is ``[3D, D]`` and a GRU's gate weights are
        ``[3H, *]``: several independent linear maps concatenated on the output
        axis, not one map.  The stack has rank at most ``D``, so orthogonalizing
        it whole flattens a *joint* spectrum and couples the blocks' scales —
        Q, K and V stop being independently conditioned.  Splitting first makes
        the result identical to holding them as separate parameters, which is
        what the reference implementations do.
        """
        if n_split == 1:
            return zeropower_via_newtonschulz5(update, steps=ns_steps)
        if update.size(-2) % n_split != 0:
            raise ValueError(
                f"split={n_split} does not divide out-features {update.size(-2)}"
            )
        rows = update.size(-2) // n_split
        return torch.cat(
            [
                zeropower_via_newtonschulz5(
                    update[i * rows:(i + 1) * rows], steps=ns_steps
                )
                for i in range(n_split)
            ],
            dim=-2,
        )

    def _step_adamw(self, group) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            exp_avg.lerp_(g, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

            bias1 = 1.0 - beta1 ** t
            bias2 = 1.0 - beta2 ** t
            denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / bias1)
