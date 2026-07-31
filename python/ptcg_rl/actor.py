"""Masked autoregressive sampling and log-probabilities — RL_SPEC §5.

**One action is a sequence, not a token.** A decision point takes
``minCount ≤ k ≤ maxCount`` picks over ≤ ``O_MAX`` options, chosen
autoregressively with a STOP column, so

```
log π_θ(a | s) = Σ_{t=1..k+1} log softmax( mask( logits_t ) )[a_t]
```

and the PPO ratio is on that **joint** sequence.  Three things follow, each of
which is a silent bug if missed:

* The sum **includes the STOP step** wherever a STOP column exists.  Dropping it
  makes short selections systematically cheaper and biases the policy toward
  tiny picks.
* Rows that declined an optional select (``minCount == 0``, no STOP column)
  contribute an **empty sum** — zero, not ``-inf``.
* Per-step ratios optimise a different objective than the joint ratio.  Never
  average them.

Everything that masks option logits goes through :func:`mask_step_logits` in this
module.  Rollout, the PPO update, the KL anchor and the gate all call it.  Three
copies of this logic would drift, and the drift is invisible until the policy
proposes an illegal pick in a live game.

Parity with IL is a hard requirement: :func:`recompute_logp` must reproduce
``ptcg_il.model.policy.multiselect_ce`` at ``label_smoothing=0``, since that is
what π_IL was fitted with and what the KL anchor compares against.
``python/tests/test_rl_actor.py`` asserts it on real shard data.
"""

from __future__ import annotations

import torch

from ptcg_il.model.policy import Policy

# Masked-out options get -inf, not -1e9.  `multiselect_ce` masks with -inf via
# `masked_label_smoothed_ce`, and log-probs from -1e9 differ from log-probs from
# -inf in the last bits — enough to break the epoch-0 ratio canary (§7), which
# is calibrated at 1e-3.
NEG_INF = float("-inf")

# The option tensors `PointerHead.forward` reads out of the featurizer dict.
# `recompute_logp` subsets these per AR step, so this list must stay in step
# with pointer.py: a key missing here raises there rather than scoring wrong.
_POINTER_KEYS = (
    "opt_type",
    "opt_src_idx",
    "opt_tgt_idx",
    "opt_card_feat",
    "opt_attack_feat",
    "opt_scalar",
    "opt_mask",
)


def mask_step_logits(
    logits: torch.Tensor,
    picked_mask: torch.Tensor,
    t: int,
    min_count: torch.Tensor,
    stop_column: torch.Tensor,
) -> torch.Tensor:
    """Legal-option mask for AR step *t*.  The single source of truth.

    Parameters
    ----------
    logits : Tensor[B, O]
        Raw pointer-head logits.
    picked_mask : bool Tensor[B, O]
        ``True`` = still selectable.  Starts as ``opt_mask`` and has each chosen
        regular option cleared; the STOP column is never cleared, because a
        sample may reach it at any later step.
    t : int
        Current AR step.
    min_count : int64 Tensor[B]
    stop_column : int64 Tensor[B]
        ``-1`` where the decision has no STOP column.

    Returns
    -------
    Tensor[B, O] with ``-inf`` at every illegal position.
    """
    out = logits.masked_fill(~picked_mask, NEG_INF)

    # STOP is illegal until the sample has taken minCount picks — otherwise the
    # policy can decline a mandatory select, which the engine rejects.
    stop_forbidden = (torch.as_tensor(t, device=logits.device) < min_count) & (stop_column >= 0)
    if stop_forbidden.any():
        rows = torch.where(stop_forbidden)[0]
        out[rows, stop_column[rows].clamp(min=0)] = NEG_INF
    return out


def masked_log_softmax(masked_logits: torch.Tensor) -> torch.Tensor:
    """``log_softmax`` in fp32 over already-masked logits.

    fp32 regardless of the surrounding autocast: §7 measured that bf16 rounds
    near-equal logits to *exactly* equal, and the resulting log-prob tail
    (p99 ≈ ln 2, max ≈ ln 8) consumes 500% of the ε = 0.2 clip range.  Upcasting
    the reduction is not a *fix* — the error is born in the network's bf16
    matmuls — but it is free and it removes one of the two sources.
    """
    return torch.log_softmax(masked_logits.float(), dim=-1)


def masked_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """``−Σ p log p`` over legal options only.

    ``log_probs`` carries ``-inf`` at masked positions, where ``p == 0``.  The
    product ``0 · -inf`` is NaN, so those terms must be neutralised — but the
    neutralisation has to happen **before** the multiply, not after.

    ``torch.where(finite, p * lp, 0)`` produces the right *value* and a NaN
    *gradient*: `where` evaluates both branches, and autograd propagates through
    the discarded one. Since entropy is part of the loss, that NaN reaches every
    parameter. Zeroing the ``-inf`` first keeps both the value and the gradient
    finite.
    """
    probs = log_probs.exp()
    safe_log = log_probs.masked_fill(~torch.isfinite(log_probs), 0.0)
    return -(probs * safe_log).sum(dim=-1)


def _ar_state(x: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(picked_mask, min_count, stop_column)`` for an AR rollout over *x*."""
    B = x["tok_type"].shape[0]
    device = x["tok_type"].device
    stop_column = x.get("stop_column")
    if stop_column is None:
        stop_column = torch.full((B,), -1, dtype=torch.long, device=device)
    return x["opt_mask"].clone(), x["minCount"], stop_column


def _advance(
    policy: Policy,
    o: torch.Tensor,
    picked_mask: torch.Tensor,
    msgru_h: torch.Tensor,
    rows: torch.Tensor,
    choice: torch.Tensor,
    stop_column: torch.Tensor,
) -> None:
    """Apply one chosen pick per row: GRU update, then clear the option.

    Mutates *picked_mask* and *msgru_h* in place, exactly as ``multiselect_ce``
    and ``_select_multi_raw`` do.  The STOP column is deliberately **not**
    cleared — a sample that could stop at step *t* can still stop at *t+1*.

    *o* is aligned to *rows*: ``o[i]`` holds the option representations for
    global row ``rows[i]``.  Callers that ran the pointer over the whole batch
    pass ``o[rows]``.  *picked_mask*, *msgru_h* and *stop_column* stay
    full-batch and are indexed by *rows*.
    """
    if rows.numel() == 0:
        return
    device = msgru_h.device
    amp_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    local = torch.arange(rows.numel(), device=o.device)
    # The multi-select GRU runs in fp32 outside autocast, matching IL.
    with torch.amp.autocast(amp_device, enabled=False):
        msgru_h[rows] = policy.pointer.msgru(o[local, choice].float(), msgru_h[rows])

    is_regular = choice != stop_column[rows]
    if is_regular.any():
        reg = rows[is_regular]
        picked_mask[reg, choice[is_regular]] = False


def recompute_logp(
    policy: Policy,
    x: dict[str, torch.Tensor],
    actions: torch.Tensor | None = None,
    action_len: torch.Tensor | None = None,
    *,
    history_h: torch.Tensor | None = None,
    encoded: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-force an action sequence under the current θ.

    Parameters
    ----------
    policy : Policy
    x : dict
        Featurizer tensor dict.
    actions : int64 Tensor[B, T] or None
        Stored picks, ``-1``-padded.  Defaults to ``x["action_idx"]``, which is
        what makes this a drop-in check against ``multiselect_ce``.
    action_len : int64 Tensor[B] or None
        Picks per row.  Defaults to ``x["action_len"]``.  Steps at or beyond a
        row's length contribute nothing.
    history_h : Tensor[B, D] or None
        Cross-turn GRU state.
    encoded : Tensor[B, L, D] or None
        A state already run through :meth:`Policy._encode`.  Callers that also
        need the value head — the PPO update and the rollout actor both do —
        should encode once and pass it here; otherwise the transformer runs
        twice per batch, doubling the cost of every update.

    Returns
    -------
    logp : fp32 Tensor[B]
        Joint log-probability of the sequence, STOP step included.  Rows with
        no picks return ``0.0`` — an empty sum, per §5.
    entropy : fp32 Tensor[B]
        Per-step masked entropy, summed over the same steps.
    """
    if actions is None:
        actions = x["action_idx"]
    if action_len is None:
        action_len = x["action_len"]

    B = x["tok_type"].shape[0]
    device = x["tok_type"].device
    h = encoded if encoded is not None else policy._encode(x, history_h)[0]
    picked_mask, min_count, stop_column = _ar_state(x)

    msgru_h = torch.zeros(B, policy.D, device=device, dtype=torch.float32)
    logp = torch.zeros(B, device=device, dtype=torch.float32)
    entropy = torch.zeros(B, device=device, dtype=torch.float32)

    n_steps = int(action_len.max().item()) if B else 0
    for t in range(min(n_steps, actions.shape[1])):
        # Select the active rows *before* scoring, and run the pointer on those
        # rows only.  This is the one AR loop that runs under `enable_grad`, so
        # every step's pointer activations — B x O_MAX x 4D — are retained for
        # backward.  Scoring the full batch at every step made that
        # `max(action_len)` times larger than the work requires, and one long
        # multi-select row set that factor for the whole minibatch.  With 94.4%
        # of real decisions taking a single pick, a 1024-row PPO minibatch spent
        # ~30x the memory it needed and OOM'd a 16 GiB card.
        target = actions[:, t]
        valid = target >= 0
        if not valid.any():
            # `continue`, not `break`: this makes no assumption that the -1
            # padding in `actions` is right-aligned.
            continue
        rows = torch.where(valid)[0]
        choice = target[rows]

        xs = {k: x[k][rows] for k in _POINTER_KEYS}
        logits, o = policy.pointer(
            h[rows], x["tok_mask"][rows], policy.embed.card, xs, msgru_h=msgru_h[rows]
        )
        masked = mask_step_logits(
            logits, picked_mask[rows], t, min_count[rows], stop_column[rows]
        )
        log_probs = masked_log_softmax(masked)

        local = torch.arange(rows.numel(), device=device)
        logp[rows] = logp[rows] + log_probs[local, choice]
        entropy[rows] = entropy[rows] + masked_entropy(log_probs)

        _advance(policy, o, picked_mask, msgru_h, rows, choice, stop_column)

    return logp, entropy


@torch.no_grad()
def sample_action(
    policy: Policy,
    x: dict[str, torch.Tensor],
    *,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    greedy: bool = False,
    history_h: torch.Tensor | None = None,
    encoded: torch.Tensor | None = None,
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
    """Sample one action sequence per row from ``π_θ``, with masking.

    Acting always samples from ``π_θ`` — never from a search-improved
    distribution.  PPO's ratio presumes ``a ~ π_θ_old``; if something else chose
    the action, the ratio is an importance weight for nothing and the update is
    biased in a way no tuning fixes (§8.2).

    Parameters
    ----------
    temperature : float
        Applied to the *masked* logits.  Masking first keeps illegal options at
        zero probability for any temperature.
    greedy : bool
        Take the argmax instead of sampling.  This is the deployment-time and
        gate-time procedure (§10.2), so the gate measures the artifact that
        ships.
    generator : torch.Generator or None
        For reproducible paired games.
    encoded : Tensor[B, L, D] or None
        A pre-encoded state, so a caller that also wants the value head does not
        run the transformer twice.  See :func:`recompute_logp`.

    Returns
    -------
    picks : list of list of int
        Per row, the chosen option indices **excluding** the STOP pick — that is
        what the engine's ``Select`` takes.
    logp : fp32 Tensor[B]
        Joint log-probability *including* the STOP step.
    entropy : fp32 Tensor[B]
        Summed masked entropy.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if generator is not None and generator.device.type != x["tok_type"].device.type:
        raise ValueError(
            f"generator is on {generator.device} but the batch is on "
            f"{x['tok_type'].device}; torch.multinomial rejects the mismatch "
            f"rather than falling back, so build the generator on the batch's device"
        )

    B = x["tok_type"].shape[0]
    device = x["tok_type"].device
    h = encoded if encoded is not None else policy._encode(x, history_h)[0]
    picked_mask, min_count, stop_column = _ar_state(x)
    max_count = x["maxCount"]

    msgru_h = torch.zeros(B, policy.D, device=device, dtype=torch.float32)
    logp = torch.zeros(B, device=device, dtype=torch.float32)
    entropy = torch.zeros(B, device=device, dtype=torch.float32)
    picks: list[list[int]] = [[] for _ in range(B)]

    active = torch.ones(B, dtype=torch.bool, device=device)
    n_steps = int(max_count.max().item()) if B else 0

    for t in range(n_steps):
        # A row is done once it has chosen STOP or exhausted its own maxCount.
        still = active & (torch.as_tensor(t, device=device) < max_count)
        if not still.any():
            break
        rows = torch.where(still)[0]

        logits, o = policy.pointer(h, x["tok_mask"], policy.embed.card, x, msgru_h=msgru_h)
        masked = mask_step_logits(logits, picked_mask, t, min_count, stop_column)
        log_probs = masked_log_softmax(masked)

        if greedy:
            choice = log_probs[rows].argmax(dim=-1)
        else:
            # Temperature is applied to the masked logits so -inf stays -inf.
            scaled = masked_log_softmax(masked[rows] / temperature)
            choice = torch.multinomial(scaled.exp(), 1, generator=generator).squeeze(-1)

        # The score is always the untempered distribution: PPO's ratio must be
        # over the density the policy actually defines, not a tempered proposal.
        logp[rows] = logp[rows] + log_probs[rows, choice]
        entropy[rows] = entropy[rows] + masked_entropy(log_probs[rows])

        chose_stop = (choice == stop_column[rows]) & (stop_column[rows] >= 0)
        for i, r in enumerate(rows.tolist()):
            if not chose_stop[i]:
                picks[r].append(int(choice[i]))
        if chose_stop.any():
            active[rows[chose_stop]] = False

        # The pointer ran over the whole batch here (no_grad, so the activations
        # do not accumulate); `_advance` wants `o` aligned to `rows`.
        _advance(policy, o[rows], picked_mask, msgru_h, rows, choice, stop_column)

    return picks, logp, entropy


def kl_to_reference(
    logp_theta: torch.Tensor,
    logp_ref: torch.Tensor,
) -> torch.Tensor:
    """Sample estimate of ``KL(π_θ ‖ π_IL)`` on the actions actually taken.

    This is the single-sample estimator ``log π_θ(a) − log π_IL(a)`` with
    ``a ~ π_θ``, which is unbiased for the mode-seeking direction RL_SPEC §9.3
    specifies.  Mode-seeking is the right direction for an *anchor*: it penalises
    θ for putting mass where IL has none.  The reverse, ``KL(π_IL ‖ π_θ)``, would
    force θ to cover all of IL's mass including its mistakes — and this IL policy
    is well short of perfect.

    Both arguments must come from the same legal mask and the **same precision**
    (§7); comparing across dtypes measures rounding, not divergence.
    """
    if logp_theta.shape != logp_ref.shape:
        raise ValueError(
            f"log-prob shapes differ ({tuple(logp_theta.shape)} vs "
            f"{tuple(logp_ref.shape)}); the two policies were not evaluated on "
            f"the same decision points"
        )
    return logp_theta - logp_ref
