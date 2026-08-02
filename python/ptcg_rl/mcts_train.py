"""AlphaZero-style MCTS self-play training on an IL-warmstarted policy.

Usage::

    cd python
    uv run python -m ptcg_rl.mcts_train \\
        --il-ckpt checkpoints_a0/ckpt-best.pt \\
        --data-dir data \\
        --out-dir checkpoints_a0_mcts

Flow:
1. Load IL checkpoint as θ_init
2. Self-play games: our deck vs sampled 𝒟_opp archetype (6 known)
3. ρ=5% of decisions get MCTS targets (π̃, Ṽ) — using the KNOWN opponent deck
4. Store (state, π̃, game outcome) in replay buffer
5. Train: CE to π̃ + MSE to outcome
6. Periodically evaluate against frozen θ_init
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(show_time=False)],
)
logger = logging.getLogger(__name__)


def fmt_dur(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptcg_rl.mcts_train",
        description="AlphaZero-style MCTS self-play training (Phase 3)",
    )
    p.add_argument("--il-ckpt", type=str, required=True,
                   help="Frozen IL policy: the KL anchor, the IL_baseline "
                        "league opponent, and θ_init unless --init-ckpt "
                        "overrides the last of those")
    p.add_argument("--init-ckpt", type=str, default=None,
                   help="Warm-start θ from this checkpoint instead of "
                        "--il-ckpt, leaving --il-ckpt as the anchor and "
                        "baseline.  Splitting them is what lets a broadly "
                        "pre-trained generalist be fine-tuned on one deck "
                        "while still being gated against that deck's "
                        "specialist.  Ignored when --resume or an "
                        "auto-resumed champion supplies θ.")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Directory with vocab.json, archetypes.json, static tables")
    p.add_argument("--out-dir", type=str, default="checkpoints_mcts",
                   help="Where MCTS-trained checkpoints are written")
    p.add_argument("--deck-archetype", type=int, default=None,
                   help="Archetype id whose decklist θ plays, and the deck "
                        "every champion written here is labelled with.  A "
                        "specialist --il-ckpt must match it: it only ever saw "
                        "that deck's cards.  Omitted = the global fixed_deck "
                        "from archetypes.json.")

    op = p.add_argument_group("Seat-1 opponent pilots")
    op.add_argument("--opp-default-ckpt", type=str, default=None,
                    help="Policy that pilots seat 1 for any sampled opponent "
                         "deck with no registered expert.  Normally the "
                         "generalist.  Without this (and without "
                         "--opp-expert) θ answers both seats, which means the "
                         "opponent plays a 60-card archetype θ has never seen "
                         "— the win rate then measures its incompetence, not "
                         "θ's edge.")
    op.add_argument("--opp-expert", action="append", default=None,
                    metavar="ID=PATH",
                    help="Repeatable ``archetype_id=checkpoint`` — the model "
                         "that owns that deck pilots seat 1 whenever it is "
                         "sampled.  Overrides --opp-default-ckpt for that id.")
    op.add_argument("--opp-expert-share", type=float, default=0.0,
                    help="Fraction of games whose opponent deck is drawn from "
                         "the --opp-expert decks instead of the whole pool.  "
                         "Left at 0 the experts only appear when their deck "
                         "happens to come up, which under --all-archetypes is "
                         "about 2%% of games (4 rivals in a 201-deck pool) — "
                         "not the round-robin it looks like.  The remaining "
                         "share still draws from the full pool, because the "
                         "live opponent is out-of-distribution and a policy "
                         "trained only against 5 known decks overfits that "
                         "metagame.  Ignored when no --opp-expert is given.")

    hp = p.add_argument_group("MCTS hyperparameters")
    hp.add_argument("--iterations", type=int, default=64,
                    help="PUCT iterations per tree")
    hp.add_argument("--c-puct", type=float, default=2.0,
                    help="PUCT exploration constant")
    hp.add_argument("--k-determinizations", type=int, default=4,
                    help="Determinizations per root (lower than training since opp deck is known)")
    hp.add_argument("--leaf-batch", type=int, default=512,
                    help="Max leaves per MCTS batch step")
    hp.add_argument("--rho", type=float, default=0.05,
                    help="Fraction of decisions tagged for MCTS")
    hp.add_argument("--mcts-distill", action="store_true", default=False,
                    help="Enable MCTS PUCT search distillation (ρ=5%% of decisions)")
    hp.add_argument("--mcts-enabled", action="store_true", default=False,
                    help=argparse.SUPPRESS)  # alias for --mcts-distill
    hp.add_argument("--n-engines", type=int, default=8,
                    help="libcg instances for MCTS forest")
    hp.add_argument("--n-workers", type=int, default=12,
                    help="RolloutPool parallel workers (libcg battles)")
    hp.add_argument("--forward-batch", type=int, default=512,
                    help="GPU forward batch size for rollout + MCTS")

    rp = p.add_argument_group("Replay buffer")
    rp.add_argument("--buffer-capacity", type=int, default=100_000,
                    help="Max decision points in replay buffer")
    rp.add_argument("--min-buffer", type=int, default=10_000,
                    help="Start training only after this many samples")

    tp = p.add_argument_group("Training")
    tp.add_argument("--total-games", type=int, default=2_000,
                    help="Total self-play games to run")
    tp.add_argument("--games-per-iter", type=int, default=50,
                    help="Self-play games per iteration before training")
    tp.add_argument("--train-steps-per-iter", type=int, default=200,
                    help="Training steps per iteration")
    tp.add_argument("--batch-size", type=int, default=256)
    tp.add_argument("--grad-accum", type=int, default=1,
                    help="Split each training step into this many "
                         "microbatches and step once, keeping the effective "
                         "batch at --batch-size.  Activations are ~99%% of "
                         "this step's memory, so N here cuts peak GPU memory "
                         "by roughly N.  Costs a little speed, and advantages "
                         "get normalised per microbatch rather than per "
                         "batch.")
    tp.add_argument("--all-archetypes", action="store_true", default=False,
                    help="Train against ALL 179 archetypes (default: 6 𝒟_opp only).")
    tp.add_argument("--lr", type=float, default=1e-4,
                    help="Peak learning rate")
    tp.add_argument("--min-lr", type=float, default=1e-5,
                    help="Minimum learning rate (cosine floor)")
    tp.add_argument("--warmup-steps", type=int, default=500,
                    help="Linear warmup steps at start of training")
    tp.add_argument("--lr-schedule", type=str, default="cosine",
                    choices=["cosine", "constant"],
                    help="LR schedule: cosine decay or constant")
    tp.add_argument("--c-value", type=float, default=0.5,
                    help="Value loss weight")
    tp.add_argument("--c-pi", type=float, default=1.0,
                    help="Policy (CE to π̃) loss weight")
    tp.add_argument("--beta", type=float, default=0.1,
                    help="KL-anchor weight on k3(π_θ ‖ π_IL). 0 disables the "
                         "anchor; the term is a real gradient, not a log line")
    tp.add_argument("--grad-clip", type=float, default=30.0,
                    help="Global grad-norm clip.  Meant as a spike guard, not "
                         "a per-step normaliser: measured raw norms on a real "
                         "run were 4.87-34.93 (median 12.68), so the old 0.5 "
                         "bound on 100%% of steps and rescaled each by a "
                         "different 10-70x factor — which inverts loss scale "
                         "rather than preserving it, since the strongest "
                         "batches get divided hardest.  Watch "
                         "train/grad_clip_frac and keep it near 0.05-0.10.")

    ep = p.add_argument_group("Evaluation & League gate")
    ep.add_argument("--eval-games", type=int, default=100,
                    help="Games per opponent in each league evaluation")
    ep.add_argument("--eval-every-games", type=int, default=250,
                    help="Evaluate every N self-play games")
    ep.add_argument("--gate-score", type=float, default=0.70,
                    help="Win rate required against EVERY past champion to save")
    ep.add_argument("--max-champions", type=int, default=10,
                    help="Max past champion checkpoints to keep and test against")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from an MCTS checkpoint .pt file")

    wb = p.add_argument_group("W&B logging")
    wb.add_argument("--wandb", action="store_true", default=False,
                    help="Log to Weights & Biases")
    wb.add_argument("--wandb-project", type=str, default="pokemon-tcg-mcts")
    wb.add_argument("--wandb-entity", type=str, default="poken")
    wb.add_argument("--wandb-name", type=str, default=None)
    wb.add_argument("--wandb-mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])
    wb.add_argument("--no-wandb", action="store_true",
                    help="Disable wandb (overrides --wandb)")
    return p


# CLI dest → the attribute name the MCTS code paths actually read.  The
# parser spells these flags without the `mcts_` prefix (`--rho`,
# `--iterations`, …) while `run_self_play_games` and `ptcg_rl.search` read
# `config.mcts_rho`, `config.mcts_iterations`, … off the same Namespace.
# Every one of those reads is a `getattr(..., default)`, so an unmapped flag
# silently took the default instead of raising.
_MCTS_ARG_ALIASES = {
    "rho": "mcts_rho",
    "iterations": "mcts_iterations",
    "c_puct": "mcts_c_puct",
    "k_determinizations": "mcts_k_determinizations",
    "leaf_batch": "mcts_leaf_batch",
    "n_engines": "mcts_n_engines",
}


def normalize_mcts_args(args: argparse.Namespace) -> argparse.Namespace:
    """Mirror the MCTS CLI flags onto the ``mcts_*`` names the code reads.

    Mutates and returns *args*.  Also unifies ``--mcts-distill`` with its
    hidden ``--mcts-enabled`` alias: ``mcts_train`` gates on the former,
    ``ptcg_rl.search`` on the latter, so either flag must set both.
    """
    for src, dst in _MCTS_ARG_ALIASES.items():
        if hasattr(args, src):
            setattr(args, dst, getattr(args, src))
    enabled = bool(getattr(args, "mcts_distill", False)) or \
        bool(getattr(args, "mcts_enabled", False))
    args.mcts_distill = enabled
    args.mcts_enabled = enabled
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The only supported way to build this CLI's config — parsing without
    the alias pass yields a Namespace on which every ``mcts_*`` read falls
    back to its default."""
    return normalize_mcts_args(build_parser().parse_args(argv))


def _load_vocab(data_dir: Path) -> dict:
    from ptcg_il.featurizer import normalize_vocab
    with open(data_dir / "vocab.json") as f:
        return normalize_vocab(json.load(f))


def _load_archetypes(data_dir: Path) -> dict:
    with open(data_dir / "archetypes.json") as f:
        return json.load(f)


def _static_tables(data_dir: Path) -> dict:
    out: dict[str, Any] = {}
    for key, fname in (("card_static_table", "card_static_table.npy"),
                       ("attack_static_table", "attack_static_table.npy")):
        path = data_dir / fname
        out[key] = torch.from_numpy(np.load(path)).float() if path.exists() else None
    return out


def _load_all_card_feat(data_dir: Path) -> torch.Tensor | None:
    """Load [n_all_cards, F_CARD] feature matrix for BeliefHeads."""
    from ptcg_il.model.cards import F_CARD

    ecf_path = data_dir / "engine_card_features.npy"
    if not ecf_path.exists():
        return None
    import numpy as _np
    ecf = _np.load(ecf_path, allow_pickle=True).item()
    if not ecf:
        return None
    max_id = max(ecf.keys())
    all_feat = torch.zeros(max_id + 1, F_CARD)
    for cid, feat in ecf.items():
        all_feat[int(cid)] = torch.from_numpy(_np.asarray(feat, dtype=_np.float32))
    return all_feat


def _load_policy(ckpt_path: str, data_dir: Path, device: torch.device) -> Any:
    """Load policy from a checkpoint (pure-feature format).

    Loads ``all_card_feat`` from the data directory for belief heads.
    """
    return _load_policy_and_deck(ckpt_path, data_dir, device)[0]


def _load_policy_and_deck(
    ckpt_path: str, data_dir: Path, device: torch.device,
) -> tuple[Any, dict | None]:
    """:func:`_load_policy`, plus the checkpoint's ``deck`` record.

    League evaluation needs both, and the record is only readable from the
    ``.pt`` — reading it back with a second :func:`torch.load` would re-read a
    226 MB generalist checkpoint per opponent per eval round.  The record is
    ``None`` for a checkpoint written before ``save_checkpoint`` stamped it.
    """
    from ptcg_il.deck import DECK_KEY
    from ptcg_il.model.policy import load_policy_state, policy_from_config
    from ptcg_il.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(ckpt_path, device="cpu")
    config = ckpt.get("config") or {}
    if not config:
        raise ValueError(f"{ckpt_path} has no 'config' record")

    all_card_feat = _load_all_card_feat(data_dir)

    policy = policy_from_config(config, all_card_feat=all_card_feat)
    model_sd = policy.state_dict()
    ckpt_sd = ckpt["model_state_dict"]
    try:
        load_policy_state(policy, ckpt_sd)
    except RuntimeError:
        _load_lenient(policy, model_sd, ckpt_sd)

    policy.to(device)
    return policy, ckpt.get(DECK_KEY)


def _load_lenient(policy: Any, model_sd: dict, ckpt_sd: dict) -> None:
    """Load *ckpt_sd* into *policy*, handling size-mismatched ``*.static`` params.

    For any parameter whose checkpoint shape differs from the model, only
    the overlapping prefix is copied.  Everything else is loaded as-is.
    Missing belief-head keys are tolerated (pre-belief checkpoints).
    """
    import torch

    loaded = 0
    skipped = 0
    for key, model_param in model_sd.items():
        ckpt_param = ckpt_sd.get(key)
        if ckpt_param is None:
            if key.startswith("belief_heads."):
                skipped += 1
                continue
            logger.warning("  _load_lenient: missing key %s", key)
            skipped += 1
            continue
        if ckpt_param.shape != model_param.shape:
            # Copy overlapping prefix (handles vocab growth)
            slices = tuple(
                slice(0, min(cs, ms))
                for cs, ms in zip(ckpt_param.shape, model_param.shape)
            )
            model_param[slices].copy_(ckpt_param[slices])
            loaded += 1
            logger.debug("  _load_lenient: %s %s → %s (partial)",
                         key, list(ckpt_param.shape), list(model_param.shape))
        else:
            model_param.copy_(ckpt_param)
            loaded += 1
    logger.info(
        "  Lenient load: %d params loaded (%d skipped) from old checkpoint",
        loaded, skipped,
    )


def _load_frozen_anchor(ckpt_path: str, data_dir: Path, device: torch.device) -> Any:
    """Load a frozen copy for evaluation."""
    from ptcg_il.model.policy import policy_from_config
    from ptcg_il.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(ckpt_path, device="cpu")
    config = ckpt.get("config") or {}
    config = dict(config)
    all_card_feat = _load_all_card_feat(data_dir)
    policy = policy_from_config(config, all_card_feat=all_card_feat)
    _load_lenient(policy, policy.state_dict(), ckpt["model_state_dict"])
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def _deck_record_of(ckpt_path: str) -> dict | None:
    """Read just the ``deck`` record off a checkpoint.

    For the frozen anchor, which :func:`_load_frozen_anchor` has already
    loaded by the time the league is assembled.  Everything else comes
    through :func:`_load_policy_and_deck` and pays no second read.
    """
    from ptcg_il.deck import DECK_KEY
    from ptcg_il.train.checkpoint import load_checkpoint

    try:
        return load_checkpoint(ckpt_path, device="cpu").get(DECK_KEY)
    except Exception as e:  # noqa: BLE001 — an unreadable record is not fatal
        logger.warning("Could not read the deck record from %s: %s",
                       ckpt_path, e)
        return None


def _opponent_deck(
    deck_record: dict | None, theta_deck: list[int],
) -> list[int] | None:
    """The decklist a league opponent brings.  ``None`` means *drop it*.

    Evaluation is best-deck vs best-deck: θ plays ``--deck-archetype``'s list
    and every opponent plays the one its own checkpoint was trained on, read
    from the ``deck`` record rather than from an archetype id (ids are cluster
    indices and get reassigned when mining re-runs; the stored decklist is
    already engine card ids, so it cannot go stale the same way).

    A *generalist* has no best deck.  Its record carries ``archetypes.json``'s
    global ``fixed_deck`` — one specific archetype's list, 21 in this corpus —
    which it is no better at than any other.  It gets θ's deck instead, making
    that pairing a mirror so ``eval/wr_vs_il`` measures the policy and not the
    deck.

    ``None`` for a checkpoint with no usable record.  The caller drops that
    opponent rather than guessing: the previous code drew a *random* deck from
    the 201-deck pool here (a fresh one per seat, per eval round), so
    iteration-to-iteration eval deltas tracked which two decks came up rather
    than anything about θ.
    """
    if not isinstance(deck_record, dict):
        return None
    # A populated `deck` is what makes a checkpoint deck-labelled at all, so it
    # is required before either branch — not just before the specialist one.
    # Reading `specialist` first would take a record carrying nothing but
    # ``{}`` for a generalist and mirror it, which looks like a deliberate
    # mirror match in the log and is really a missing label.
    deck = deck_record.get("deck")
    if not isinstance(deck, list) or not deck:
        return None
    if not deck_record.get("specialist"):
        return list(theta_deck)
    return [int(c) for c in deck]


def _fixed_deck(data_dir: Path) -> list[int]:
    with open(data_dir / "archetypes.json") as f:
        return [int(c) for c in json.load(f)["fixed_deck"]]


def _deck_for(data_dir: Path, deck_archetype: int | None) -> list[int]:
    """The 60-card decklist θ plays.

    ``archetypes.json``'s ``fixed_deck`` is one specific archetype's
    representative (the best 𝒟_self one), so using it for *every* specialist
    would hand, say, the a0 model a deck it has never seen.  That fails
    silently — cards outside the model's vocab just become all-zero feature
    rows — which is why the deck has to follow the checkpoint, exactly as
    ``ptcg_rl.train`` does it.

    ``None`` keeps the historical behaviour (the global ``fixed_deck``).
    """
    if deck_archetype is None:
        return _fixed_deck(data_dir)
    from ptcg_il.deck import build_deck_metadata

    deck = build_deck_metadata(data_dir, deck_archetype)
    decklist = deck.get("deck")
    if not isinstance(decklist, list) or not decklist:
        raise ValueError(
            f"archetype {deck_archetype} produced no decklist from {data_dir}"
        )
    return [int(c) for c in decklist]


def _check_ckpt_deck(il_ckpt: str, deck_archetype: int | None) -> None:
    """Fail loudly when *il_ckpt* was trained on a different archetype.

    ``save_checkpoint`` stamps the deck record precisely because this pairing
    is otherwise invisible at runtime.

    A *generalist* record (``archetype_self`` is ``None``) is a legitimate
    warm-start for any deck in its training mix — that is the whole point of
    pre-training on every archetype — so it is allowed through.  It used to be
    allowed through by accident, because ``None`` failed the mismatch test and
    fell out of the bottom, which meant a generalist trained on a *different*
    corpus passed just as quietly.  Say so explicitly instead, and check the
    thing that actually matters for a generalist: that the deck's cards were
    in its vocabulary.
    """
    if deck_archetype is None:
        return
    try:
        blob = torch.load(il_ckpt, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 — the real load downstream reports properly
        return
    record = blob.get("deck") if isinstance(blob, dict) else None
    if not isinstance(record, dict):
        logger.warning(
            "%s carries no deck record — cannot verify it was trained on "
            "archetype %d", il_ckpt, deck_archetype,
        )
        return
    trained_on = record.get("archetype_self")
    if trained_on is None:
        logger.info(
            "%s is a generalist (archetype_self=None) — warm-starting deck "
            "%d from it.  Its coverage of that deck rests on the corpus it "
            "was trained over, which %s pins.",
            il_ckpt, deck_archetype, record.get("archetypes_sha1", "?")[:12],
        )
        return
    if int(trained_on) != int(deck_archetype):
        raise SystemExit(
            f"--deck-archetype {deck_archetype} does not match --il-ckpt "
            f"{il_ckpt}, which was trained on archetype {trained_on}.  The "
            f"specialist has never seen this deck's cards; training it here "
            f"would silently feed it all-zero card features."
        )


def _deck_record_for(data_dir: Path, deck_archetype: int | None) -> dict:
    """The deck label stamped onto every champion written by this run.

    Built from *deck_archetype* — the same input :func:`_deck_for` uses to
    pick the decklist θ actually plays — and deliberately *not* copied from
    the IL checkpoint.  The checkpoint's record describes what that model was
    trained on, which is a different question and a different answer whenever
    θ_init is a generalist: a generalist's record carries ``archetypes.json``'s
    ``fixed_deck``, so copying it labelled champions with a deck they never
    play.  Nothing downstream re-derives the label, so that reached the
    submission unchallenged.
    """
    from ptcg_il.deck import build_deck_metadata

    return build_deck_metadata(data_dir, deck_archetype)


def _discover_il_checkpoints(il_ckpt_path: str) -> list[tuple[str, str]]:
    """Discover all .pt checkpoints in the same directory as *il_ckpt_path*.

    Returns a list of (name, path) tuples.  Step checkpoints are sorted by
    step number; ``ckpt-best`` (if present) comes last.
    """
    ckpt_dir = Path(il_ckpt_path).resolve().parent
    step_ckpts = sorted(
        ckpt_dir.glob("ckpt-step-*.pt"),
        key=lambda p: int(p.stem.rsplit("-", 1)[-1]),
    )
    result: list[tuple[str, str]] = []
    for p in step_ckpts:
        result.append((p.stem, str(p)))
    best = ckpt_dir / "ckpt-best.pt"
    if best.exists():
        result.append(("ckpt-best", str(best)))
    return result


#: Engine seat that holds ``deck_self``.  ``Battle::start`` copies deck_self
#: into ``cards[..60]`` and deck_opp into ``cards[60..]`` regardless of the
#: ``our_player`` argument (ptcg_search/src/vec_env.rs), so the deck-to-seat
#: mapping is fixed and ``our_player`` only selects whose perspective the
#: reward is reported in.  θ is a specialist on ``fixed_deck``; this is the
#: seat it is competent in.
SELF_SEAT = 0


def _orient_outcome(outcome: float, your_index: int) -> float:
    """Re-express *outcome* in the frame of the player to move.

    The engine reward arrives in :data:`SELF_SEAT`'s frame.  An observation is
    egocentric, so a seat-1 decision looks exactly like a seat-0 one to the
    network — feeding it the seat-0 reward labels identical inputs ``+1`` and
    ``-1`` by turn parity, and MSE's answer to that is to predict 0.
    """
    return float(outcome) if your_index == SELF_SEAT else -float(outcome)


def _decks_by_seat(
    theta_deck: list[int], other_deck: list[int], theta_seat: int,
) -> tuple[list[int], list[int]]:
    """Order a deck pair as ``(seat 0's deck, seat 1's deck)``.

    ``RustVecEnv`` deals its ``deck_self`` argument to seat 0 and ``deck_opp``
    to seat 1 no matter what ``our_player`` says, so a caller that alternates
    seats with a fixed deck order swaps the *decks* too.  Routing the pair
    through here keeps ``theta_deck`` on ``theta_seat``, leaving turn order as
    the only thing the seat loop varies.
    """
    if theta_seat == 0:
        return theta_deck, other_deck
    return other_deck, theta_deck


def parse_opp_experts(specs: list[str] | None) -> dict[int, str]:
    """``["17=a.pt", "25=b.pt"]`` → ``{17: "a.pt", 25: "b.pt"}``.

    Raises on a malformed entry rather than skipping it.  A silently dropped
    expert costs nothing visible — that deck just falls back to the generalist
    pilot — so the branch would run to completion reporting a win rate against
    an opponent field the caller believes it configured and did not.
    """
    out: dict[int, str] = {}
    for spec in specs or []:
        arch, sep, path = str(spec).partition("=")
        if not sep or not arch.strip() or not path.strip():
            raise SystemExit(
                f"--opp-expert {spec!r} is not ID=PATH "
                "(e.g. --opp-expert 17=checkpoints_a17_mcts/ckpt-mcts-last.pt)"
            )
        try:
            aid = int(arch.strip())
        except ValueError:
            raise SystemExit(
                f"--opp-expert {spec!r}: {arch!r} is not an archetype id"
            ) from None
        if not Path(path.strip()).is_file():
            raise SystemExit(
                f"--opp-expert {spec!r}: {path.strip()} does not exist"
            )
        out[aid] = path.strip()
    return out


class OpponentExperts:
    """Seat-1 pilots, keyed by the *opponent deck's* archetype id.

    Each of the six branch decks has an owning expert; every other archetype in
    the 179-deck pool falls back to *default_ckpt* (the generalist).  Keeping
    the pool at 179 rather than narrowing it to the six is deliberate: the
    Kaggle opponent is out-of-distribution by construction, and this only
    changes *who pilots* the sampled deck, never which decks are sampled.

    Policies are loaded lazily and cached on the CPU; at most one is resident on
    the training device at a time, released when the next opponent group starts.
    Self-play already groups its games by opponent id, so that is one transfer
    per group, and it keeps the seat-1 pool from competing with θ for VRAM the
    way holding six policies would.
    """

    def __init__(
        self,
        default_ckpt: str | None,
        experts: dict[int, str],
        data_dir: Path,
        device: torch.device,
        vocab: dict,
        engine_card_features: dict | None = None,
        engine_attack_features: dict | None = None,
    ):
        self.default_ckpt = default_ckpt
        self.experts = dict(experts)
        self.data_dir = data_dir
        self.device = device
        self.vocab = vocab
        self.engine_card_features = engine_card_features
        self.engine_attack_features = engine_attack_features
        self._cache: dict[str, Any] = {}
        self._resident: str | None = None

    @property
    def enabled(self) -> bool:
        """False when no seat-1 pilot is configured at all.

        The caller passes ``act_fn_opp=None`` then, which is the historical
        one-actor-answers-both-seats path.
        """
        return bool(self.default_ckpt or self.experts)

    def ckpt_for(self, opp_id: int) -> str | None:
        return self.experts.get(int(opp_id), self.default_ckpt)

    def is_expert(self, opp_id: int) -> bool:
        """True when *opp_id* is flown by its own deck's owner.

        Distinct from :meth:`ckpt_for` returning something: the default pilot
        covers every archetype, so "has a pilot" is not "has an expert" and
        conflating them would report a 100% rival share against a field that is
        almost entirely generalist.
        """
        return int(opp_id) in self.experts

    def actor_for(self, opp_id: int, seed: int) -> Any | None:
        """A greedy :class:`PolicyActor` piloting the deck of *opp_id*.

        Greedy, not sampled: seat 1 is a fixed reference this branch is being
        measured against, and a sampling opponent adds variance to the win rate
        without making the opponent any stronger.
        """
        from ptcg_rl.rollout import PolicyActor

        path = self.ckpt_for(opp_id)
        if path is None:
            return None
        policy = self._cache.get(path)
        if policy is None:
            policy = _load_frozen_anchor(path, self.data_dir, self.device)
            self._cache[path] = policy
        if self._resident is not None and self._resident != path:
            self._cache[self._resident].to("cpu")
        policy.to(self.device)
        policy.eval()
        self._resident = path
        return PolicyActor(
            policy, self.vocab, device=str(self.device), greedy=True,
            bf16=True, seed=seed,
            engine_card_features=self.engine_card_features,
            engine_attack_features=self.engine_attack_features,
        )

    def dispatch_actor(self, opp_ids: list[int], seed: int) -> Any | None:
        """One seat-1 callable that routes each request by its ``opp_id``.

        A mixed-opponent pool has several archetypes in flight at once, so the
        pilot can no longer be chosen per pool the way it was per group.  Every
        pilot the pool can need is loaded up front and *stays* resident: the
        alternative — swapping one policy on and off the device per sub-batch —
        would run a PCIe round trip inside the hot path.  At ≤6 distinct pilots
        and 14.4M parameters that is ~350 MB, against 16 GB of card.

        Returns ``None`` when nothing is configured, which leaves
        ``collect_batch`` on its historical one-actor path.
        """
        from ptcg_rl.rollout import PolicyActor

        if not self.enabled:
            return None

        actors: dict[str, Any] = {}
        for i, oid in enumerate(sorted({int(o) for o in opp_ids})):
            path = self.ckpt_for(oid)
            if path is None:
                continue
            if path not in actors:
                policy = self._cache.get(path)
                if policy is None:
                    policy = _load_frozen_anchor(path, self.data_dir, self.device)
                    self._cache[path] = policy
                policy.to(self.device).eval()
                actors[path] = PolicyActor(
                    policy, self.vocab, device=str(self.device), greedy=True,
                    bf16=True, seed=seed + 7919 * (i + 1),
                    engine_card_features=self.engine_card_features,
                    engine_attack_features=self.engine_attack_features,
                )
        if not actors:
            return None
        self._resident = None  # every pilot is resident now; no single one owns it
        default = self.default_ckpt

        def dispatch(requests: list[dict]) -> list[dict]:
            by_path: dict[str, list[int]] = {}
            for i, req in enumerate(requests):
                path = self.ckpt_for(int(req.get("opp_id", -1))) or default
                if path not in actors:
                    # Unknown archetype and no default: fall back to any loaded
                    # pilot rather than drop the request — an unanswered
                    # request desynchronises the whole reply batch.
                    path = next(iter(actors))
                by_path.setdefault(path, []).append(i)

            replies: list[dict | None] = [None] * len(requests)
            for path, idx in by_path.items():
                sub = actors[path]([requests[i] for i in idx])
                for k, rep in enumerate(sub):
                    replies[idx[k]] = rep
            if any(r is None for r in replies):
                raise RuntimeError(
                    "opponent dispatch left a request unanswered"
                )
            return replies

        dispatch.actors = list(actors.values())  # type: ignore[attr-defined]
        return dispatch

    def release(self) -> None:
        """Move every loaded opponent off the training device."""
        for policy in self._cache.values():
            policy.to("cpu")
        self._resident = None

    def describe(self) -> str:
        if not self.enabled:
            return "θ itself (no --opp-default-ckpt / --opp-expert)"
        parts = [f"default={self.default_ckpt}" if self.default_ckpt
                 else "default=θ itself"]
        parts += [f"a{aid}={Path(p).parent.name}/{Path(p).name}"
                  for aid, p in sorted(self.experts.items())]
        return ", ".join(parts)


def run_self_play_games(
    policy: Any,
    vocab: dict,
    archetypes: dict,
    fixed_deck: list[int],
    opp_decks: list[dict],
    config: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    n_games: int,
    opp_weights: dict[int, float] | None = None,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
    opp_experts: "OpponentExperts | None" = None,
) -> tuple[list[dict], dict]:
    """Run *n_games* self-play games.  Returns (samples, timing_stats).

    If *opp_weights* is given, samples opponents with those relative
    weights (low win rate → high weight).  If None, uses uniform
    weights.  *opp_weights* is updated in-place with per-opponent
    win/loss stats.

    *opp_experts* supplies the seat-1 pilot for each sampled opponent deck.
    Without it θ answers both seats — and θ is a specialist on ``fixed_deck``,
    so it plays the opponent's archetype with cards it has never seen.  That is
    what inflates the self-play win rate; the decisions were already dropped
    from training, so only who *acts* changes here.
    """
    from ptcg_rl.rollout import PolicyActor
    from ptcg_rl.rust_vec_env import RustVecEnv

    # Derived once per call, not per forest: the table has ~1.5k entries and
    # the answer cannot change mid-batch.
    basic_pokemon_ids = _basic_pokemon_ids(engine_card_features)
    if getattr(config, "mcts_distill", False) and not basic_pokemon_ids:
        logger.warning(
            "No engine_card_features — the MCTS determinizer will guess a "
            "face-down opponent active from the whole deck template, and the "
            "engine refuses most of those roots",
        )

    all_samples: list[dict] = []
    wins, losses, draws = 0, 0, 0
    total_decisions = 0
    total_mcts_decisions = 0
    n_dropped_seat = 0   # opponent-seat decisions played but not learned from
    opp_counts: dict[int, int] = {}
    opp_wins: dict[int, int] = {}
    opp_losses: dict[int, int] = {}
    n_opp_piloted = 0            # games whose seat 1 had a real pilot
    n_expert_piloted = 0         # …of those, games flown by the deck's *owner*
    opp_pilots: dict[int, str] = {}  # opponent id → the checkpoint that flew it

    # ── Sample opponents with adaptive weights ───────────────────────
    #
    # A share of the games is drawn from the *rival* decks — the ones with a
    # registered expert — and the rest from the whole pool.  Without this the
    # experts are a rounding error: 4 rival decks in a 201-deck pool is 2% of
    # games, so "branch a1 fights the other experts" would be ~4 games in 200.
    # The remainder still draws from the full pool on purpose; the live
    # opponent is out-of-distribution, and training only against 5 known decks
    # fits a metagame that does not exist outside this loop.
    rival_decks: list[dict] = []
    expert_share = float(getattr(config, "opp_expert_share", 0.0) or 0.0)
    if opp_experts is not None and expert_share > 0.0:
        rival_decks = [d for d in opp_decks
                       if opp_experts.is_expert(int(d["id"]))]
    if expert_share > 0.0 and not rival_decks:
        # Round 1, branch 1: nobody else has a champion yet.  Say so — an
        # --opp-expert-share that silently does nothing looks identical to one
        # that is working.
        logger.info(
            "--opp-expert-share %.2f has no rival decks to draw from "
            "(no --opp-expert registered) — sampling the full pool",
            expert_share,
        )

    # The draw itself happens per *battle slot* below, not per game: a slot
    # keeps its opponent across restarts, so the sampling unit is the slot.
    # Counts are accumulated from the games that actually finished, which is
    # also the honest denominator — the old loop counted intent, and every
    # game it sampled beyond what the pool played was still in the histogram.

    t_start = time.perf_counter()
    game_i = 0
    n_workers = getattr(config, "n_workers", 12)
    perf_t_feat = 0.0; perf_t_fwd = 0.0; perf_n_actor = 0
    perf_t_sweep = 0.0; perf_t_act = 0.0; perf_n_act = 0
    perf_t_collate = 0.0; perf_t_d2h = 0.0; perf_n_rows = 0
    perf_t_record = 0.0
    _reset_mcts_perf()   # this call's search timers start from zero

    # ── Progress bar (TTY only; falls back to text logging) ──────────
    import sys as _sys
    _is_tty = _sys.stderr.isatty()
    if _is_tty:
        from rich.progress import (
            BarColumn, Progress, TaskProgressColumn, TextColumn,
            TimeElapsedColumn,
        )
        pbar = Progress(
            TextColumn("  [bold]{task.description}[/]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("• wr={task.fields[wr]:.2f}"),
            TextColumn("• {task.fields[rate]:.1f} g/min"),
            TextColumn("• {task.fields[dec_per]:.0f} d/g"),
            TimeElapsedColumn(),
        )
        task = pbar.add_task(
            f"self-play {n_games}g", total=n_games,
            wr=0.0, rate=0.0, dec_per=0.0,
        )
        pbar.start()
    else:
        pbar = None
        task = None

    # ── One pool, many opponents ──────────────────────────────────────
    #
    # Battles used to be grouped by opponent because a pool was pinned to one
    # (deck_self, deck_opp) pair.  Under --all-archetypes that gave mostly
    # one-game groups, and the two ways out both lost: sizing the pool to
    # --n-workers ran battles that were destroyed unplayed (211 actor rows per
    # recorded decision at 128 workers, throughput 19.1 → 15.5 games/min), while
    # capping it to the group size collapsed the inference batch to 1.9
    # rows/call and pushed H2D to 22% of wall.  `vec_env_create_multi` deals a
    # different opponent to each battle slot, so one wide pool is both fully
    # wanted and wide enough to batch.
    n_envs = max(1, min(n_workers, n_games))
    slot_opps: list[dict] = []
    for _ in range(n_envs):
        pool_choice = opp_decks
        if rival_decks and rng.random() < expert_share:
            pool_choice = rival_decks
        slot_opps.append(_sample_weighted(pool_choice, rng, opp_weights))
    deck_by_id = {int(d["id"]): d["deck"] for d in opp_decks}

    actor = PolicyActor(policy, vocab, device=str(device), bf16=True,
                        seed=config.seed,
                        engine_card_features=engine_card_features,
                        engine_attack_features=engine_attack_features)
    policy.eval()

    slot_ids = [int(o["id"]) for o in slot_opps]
    opp_actor = None
    if opp_experts is not None and opp_experts.enabled:
        opp_actor = opp_experts.dispatch_actor(slot_ids, config.seed)
    logger.info(
        "  pool: %d battles over %d distinct opponent archetype(s)",
        n_envs, len(set(slot_ids)),
    )

    with RustVecEnv(
        deck_self=fixed_deck,
        opp_decks=[(int(o["id"]), o["deck"]) for o in slot_opps],
        n_envs=n_envs,
        our_player=SELF_SEAT,
        seed=config.seed,
    ) as pool:
        # Drain n_games complete games.  Loop because a single collect()
        # call may return fewer games than needed if the decision budget
        # runs out before all games finish.
        remaining = n_games
        while remaining > 0 and game_i < n_games:
            # ~200 decisions/game across both seats, plus headroom.  With a
            # seat-1 pilot only θ's half is recorded, and this budget counts
            # *recorded* decisions — leaving it at 300 would make
            # collect_batch play about twice the games asked for and throw
            # the surplus away.
            per_game = 150 if opp_actor is not None else 300
            budget = remaining * per_game
            trajectories = pool.collect_batch(
                actor, budget, act_fn_opp=opp_actor)
            if not trajectories:
                break
            for traj in trajectories:
                if remaining <= 0 or game_i >= n_games:
                    break
                remaining -= 1
                game_i += 1
                t_game = time.perf_counter()
                outcome = float(traj.reward)
                decisions = traj.decisions
                # Opponent now varies per battle, so it is read off the
                # trajectory rather than being a property of the pool.
                opp_id = int(traj.opp_id)
                opp_decklist = deck_by_id.get(opp_id, [])
                seed_for = config.seed + game_i
                opp_counts[opp_id] = opp_counts.get(opp_id, 0) + 1
                if opp_actor is not None:
                    n_opp_piloted += 1
                    opp_pilots[opp_id] = opp_experts.ckpt_for(opp_id)
                    if opp_experts.is_expert(opp_id):
                        n_expert_piloted += 1

                if traj.error:
                    logger.warning(
                        "  game %d ERROR opp=%d: %s",
                        game_i, opp_id, traj.error,
                    )
                if len(decisions) < 10:
                    logger.warning(
                        "  game %d SHORT: %d dec outcome=%.0f "
                        "opp=%d timeout=%s",
                        game_i, len(decisions), outcome, opp_id,
                        traj.timeout,
                    )

                if not decisions:
                    continue

                if outcome > 0:
                    wins += 1
                    opp_wins[opp_id] = opp_wins.get(opp_id, 0) + 1
                elif outcome < 0:
                    losses += 1
                    opp_losses[opp_id] = opp_losses.get(opp_id, 0) + 1
                else:
                    draws += 1

                total_decisions += len(decisions)

                # θ answers *both* seats (RustVecEnv.collect_batch routes
                # every pending obs through one act_fn), but it is a
                # specialist on `fixed_deck`, which the engine always deals
                # to seat 0.  Training on the seat-1 decisions would ask it
                # to learn a second archetype's strategy from data it plays
                # badly, so they are driven and then dropped.
                own_idx = [i for i, d in enumerate(decisions)
                           if d.your_index == SELF_SEAT]
                n_dropped_seat += len(decisions) - len(own_idx)
                if not own_idx:
                    continue

                n_tag = max(1, int(len(own_idx)
                                   * getattr(config, "mcts_rho", 0.05)))
                local_rng = np.random.default_rng(seed_for)
                # Tag only our own seat: add_root() hands the searcher
                # `fixed_deck` and the opponent `opp_deck`, which is exactly
                # backwards for a seat-1 root.
                tagged_idx = set(
                    local_rng.choice(
                        np.asarray(own_idx),
                        size=min(n_tag, len(own_idx)), replace=False,
                    ).tolist()
                )
                mcts_targets: dict[int, tuple] = {}
                if tagged_idx and getattr(config, "mcts_distill", False):
                    mcts_targets = _run_mcts_on_decisions_pooled(
                        decisions, tagged_idx, policy, vocab,
                        fixed_deck, opp_decklist, config, device, seed_for,
                        basic_pokemon_ids=basic_pokemon_ids,
                        engine_card_features=engine_card_features,
                        engine_attack_features=engine_attack_features,
                    )
                total_mcts_decisions += len(mcts_targets)

                for i in own_idx:
                    dec = decisions[i]
                    pi, v = mcts_targets.get(i, (None, None))
                    all_samples.append({
                        "features": dec.features,
                        "action_idx": dec.action_idx,
                        "action_len": dec.action_len,
                        "logp": dec.logp,
                        "value": dec.value,
                        "mcts_pi": pi,
                        "mcts_value": v,
                        # Oriented to the player who *made* this decision.
                        # `outcome` is the engine reward in SELF_SEAT's
                        # frame; the critic must speak in the perspective
                        # of the player to move, because puct.rs
                        # expand_leaf negates it by player_role and would
                        # otherwise double-count the flip.
                        "game_outcome": _orient_outcome(
                            outcome, dec.your_index),
                        "seat": dec.your_index,
                        "opp_archetype": opp_id,
                        # Decision index within the game.  Carried so value
                        # accuracy can be read against game stage — a critic
                        # no better late than early has learned nothing, and
                        # that is invisible in a pooled correlation.
                        "turn": dec.turn,
                    })

                t_game = time.perf_counter() - t_game

                elapsed = time.perf_counter() - t_start
                if pbar is not None:
                    pbar.update(task, completed=game_i,
                                wr=wins / max(game_i, 1),
                                rate=game_i / elapsed * 60 if elapsed > 0 else 0,
                                dec_per=total_decisions / max(game_i, 1))
                elif game_i % 10 == 0:
                    logger.info(
                        "  game %d/%d | wr=%.2f %dW/%dL/%dD | "
                        "%.0f dec/game | %.1f g/min",
                        game_i, n_games,
                        wins / max(game_i, 1), wins, losses, draws,
                        total_decisions / max(game_i, 1),
                        game_i / elapsed * 60 if elapsed > 0 else 0,
                    )

        # Accumulate perf once the pool is drained — at the `with` level, not
        # inside the collect loop, or every extra collect() iteration re-adds
        # the same cumulative counters.
        #
        # The seat-1 pilot is now a *dispatcher* over several PolicyActors, one
        # per distinct opponent checkpoint; reading only `opp_actor` would miss
        # all of their featurize and forward time.
        opp_sub = list(getattr(opp_actor, "actors", []) or [])
        for a in [actor, *opp_sub]:
            if a is None or getattr(a, "perf_n_calls", 0) <= 0:
                continue
            perf_t_feat += a.perf_t_featurize
            perf_t_fwd += a.perf_t_forward
            perf_t_collate += getattr(a, "perf_t_collate", 0.0)
            perf_t_d2h += getattr(a, "perf_t_d2h", 0.0)
            perf_n_actor += a.perf_n_calls
            perf_n_rows += getattr(a, "perf_n_rows", 0)
        if pool._perf_n_act > 0:
            perf_t_sweep += pool._perf_t_sweep
            perf_t_act += pool._perf_t_act
            perf_t_record += pool._perf_t_record
            perf_n_act += pool._perf_n_act

    # ── Update adaptive weights ──────────────────────────────────────
    if opp_weights is not None:
        for d in opp_decks:
            oid = d["id"]
            w_total = opp_wins.get(oid, 0) + opp_losses.get(oid, 0)
            if w_total > 0:
                wr = opp_wins.get(oid, 0) / w_total
                # Weight ∝ (1 - win_rate) + ε → hard opponents get more samples
                opp_weights[oid] = max(0.05, 1.0 - wr)
            else:
                opp_weights[oid] = 1.0  # unknown opponent → prioritize
        # Normalize
        total_w = sum(opp_weights.values())
        if total_w > 0:
            for oid in opp_weights:
                opp_weights[oid] /= total_w

    if pbar is not None:
        pbar.stop()
    if opp_experts is not None:
        # Hand the GPU back before the training phase asks for it — the seat-1
        # pool is idle from here until the next self-play call.
        opp_experts.release()
    elapsed = time.perf_counter() - t_start
    from ptcg_rl.search import reset_leaf_perf
    leaf_perf = reset_leaf_perf()
    mcts_perf = _reset_mcts_perf()
    t_mcts = (mcts_perf["t_add_root"] + mcts_perf["t_select"]
              + mcts_perf["t_eval"] + mcts_perf["t_expand"]
              + mcts_perf["t_agg"])
    # The actor phases are nested inside t_act, which is nested inside the
    # wall clock; only the disjoint top-level slices are subtracted so
    # "other" cannot go negative by double-counting.
    perf_summary = {
        "t_featurize": perf_t_feat, "t_forward": perf_t_fwd,
        "t_collate": perf_t_collate, "t_d2h": perf_t_d2h,
        "n_actor_calls": perf_n_actor, "n_actor_rows": perf_n_rows,
        "t_sweep": perf_t_sweep, "t_act": perf_t_act,
        "t_record": perf_t_record,
        "n_act_calls": perf_n_act,
        "t_mcts": t_mcts,
        **{f"mcts_{k}": v for k, v in mcts_perf.items()},
        **{f"leaf_{k}": v for k, v in leaf_perf.items()},
        "t_other": elapsed - perf_t_act - perf_t_sweep - perf_t_record - t_mcts,
    }
    if perf_n_actor > 0 or perf_n_act > 0:
        pct = lambda v: v / elapsed * 100 if elapsed > 0 else 0.0
        logger.info(
            "PERF │ wall=%.1fs │ act=%.1fs (%.0f%%) = feat %.1f + h2d %.1f + "
            "fwd %.1f + d2h %.1f │ rust=%.1fs (%.0f%%) │ record=%.1fs │ "
            "mcts=%.1fs (%.0f%%) │ other=%.1fs (%.0f%%)",
            elapsed,
            perf_t_act, pct(perf_t_act),
            perf_t_feat, perf_t_collate, perf_t_fwd, perf_t_d2h,
            perf_t_sweep, pct(perf_t_sweep),
            perf_t_record,
            t_mcts, pct(t_mcts),
            perf_summary["t_other"], pct(perf_summary["t_other"]),
        )
        if perf_n_rows > 0:
            logger.info(
                "     │ %d actor calls, %d rows (%.1f rows/call) — "
                "feat %.2fms/row, fwd %.2fms/row",
                perf_n_actor, perf_n_rows, perf_n_rows / max(perf_n_actor, 1),
                perf_t_feat / perf_n_rows * 1000,
                perf_t_fwd / perf_n_rows * 1000,
            )
        if t_mcts > 0:
            logger.info(
                "     │ mcts: add_root %.1fs + select %.1fs + eval %.1fs "
                "+ expand %.1fs + agg %.1fs │ %.0f roots, %.0f leaves",
                mcts_perf["t_add_root"], mcts_perf["t_select"],
                mcts_perf["t_eval"], mcts_perf["t_expand"], mcts_perf["t_agg"],
                mcts_perf["n_roots"], mcts_perf["n_leaves"],
            )
            # t_eval is CPU featurize + GPU forward on two different
            # processors with opposite fixes; reported together it pointed at
            # the GPU when the cost was the serial per-leaf featurize loop.
            logger.info(
                "     │ mcts eval split: featurize %.1fs (CPU, serial) + "
                "forward %.1fs (GPU) over %.0f leaves",
                leaf_perf["t_featurize"], leaf_perf["t_forward"],
                leaf_perf["n_leaves"],
            )

    return all_samples, {
        "sp_games": n_games,
        "sp_sec": elapsed,
        "sp_games_per_min": n_games / elapsed * 60 if elapsed > 0 else 0,
        "sp_decisions": total_decisions,
        "sp_mcts_decisions": total_mcts_decisions,
        "sp_dec_per_game": total_decisions / max(n_games, 1),
        "sp_win_rate": wins / max(n_games, 1),
        "sp_wins": wins, "sp_losses": losses, "sp_draws": draws,
        "sp_opp_distribution": opp_counts,
        "sp_samples": len(all_samples),
        "sp_dropped_seat": n_dropped_seat,
        # Share of games whose seat 1 was flown by a real pilot rather than by
        # θ.  0.0 means every opponent deck was piloted by a specialist that has
        # never seen it, and the win rate above is not a measure of θ's edge.
        "sp_opp_piloted_frac": n_opp_piloted / max(n_games, 1),
        # …and of those, the share flown by a rival *expert* rather than the
        # generalist.  This is the round-robin dial: it should track
        # --opp-expert-share, and 0.0 means the branch never met another branch.
        "sp_expert_opp_frac": n_expert_piloted / max(n_games, 1),
        "sp_opp_pilots": opp_pilots,
        "perf": perf_summary,
    }


def _basic_pokemon_ids(engine_card_features: dict | None) -> list[int]:
    """Engine card ids flagged Basic, for the MCTS determinizer.

    Returns an empty list when the feature table is unavailable, which leaves
    the Rust side on its old whole-template guess rather than asserting a set
    we cannot actually justify.
    """
    from ptcg_il.featurizer import CARD_FEAT_BASIC_COL

    if not engine_card_features:
        return []
    return [
        int(cid) for cid, row in engine_card_features.items()
        if len(row) > CARD_FEAT_BASIC_COL and row[CARD_FEAT_BASIC_COL] > 0.5
    ]


#: Cumulative MCTS search phases, in seconds.  Module-level because the search
#: is called per game from deep inside the self-play loop and threading an
#: accumulator through every frame would touch six signatures for a counter.
#: `run_self_play_games` snapshots and resets it per call.
_MCTS_PERF: dict[str, float] = {
    "t_add_root": 0.0,   # Rust: build the root + determinize
    "t_select": 0.0,     # Rust: PUCT descent to leaves
    "t_eval": 0.0,       # GPU: batch_evaluate_leaves
    "t_expand": 0.0,     # Rust: write priors/values back into the trees
    "t_agg": 0.0,        # Python: aggregate K determinizations into pi-tilde
    "n_roots": 0.0,
    "n_leaves": 0.0,
}


def _reset_mcts_perf() -> dict[str, float]:
    """Snapshot and zero the search timers.  Returns the snapshot."""
    snap = dict(_MCTS_PERF)
    for k in _MCTS_PERF:
        _MCTS_PERF[k] = 0.0
    return snap


def _run_mcts_on_decisions_pooled(
    decisions: list,
    tagged_idx: set,
    policy: Any,
    vocab: dict,
    fixed_deck: list[int],
    opp_deck: list[int],
    config: Any,
    device: Any,
    seed: int,
    basic_pokemon_ids: list[int] | None = None,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> dict[int, tuple]:
    """Same as _run_mcts_on_decisions but shares the MctsForest across calls.

    The sharing is the point, not a speed trick.  This runs once per game,
    and a forest owns ``--n-engines`` libcg agents; `AgentStart` has no
    counterpart in libcg's ABI, so a pool built and freed per game strands
    its arenas for the life of the process (~50 MB/game measured at
    ``--n-engines 20``, which is what put self-play into swap around game
    2000).  `shared_forest` hands back the one process-wide pool with its
    trees cleared.
    """
    import json as _json
    from ptcg_rl.search import batch_evaluate_leaves, shared_forest

    # Phase timers for the search itself.  Nothing here was measured before, so
    # with --rho 1.0 the single most expensive stage of a distillation run was
    # invisible: `add_root` and `expand_batch` are Rust/libcg, while
    # `batch_evaluate_leaves` is another GPU forward on a *different* batch
    # shape than the rollout's, and they need separating to know which to cut.
    _mp = _MCTS_PERF
    _t_phase = time.perf_counter()

    forest = shared_forest(
        n_engines=getattr(config, "mcts_n_engines", 4),
        libcg_path=None,
    )
    if basic_pokemon_ids:
        forest.set_basic_pokemon(basic_pokemon_ids)
    tree_map: dict[int, tuple[int, int]] = {}
    K = getattr(config, "mcts_k_determinizations", 4)
    n_attempted = 0
    n_rejected = 0

    try:
        for i in tagged_idx:
            dec = decisions[i]
            obs_json = getattr(dec, "obs_json", None)
            if not obs_json:
                continue
            obs_dict = _json.loads(obs_json)
            if not obs_dict.get("search_begin_input"):
                # libcg's SearchBegin dereferences the pointer unconditionally;
                # an empty search_begin_input takes down the whole process, so
                # drop the root here rather than learn about it from a SIGSEGV.
                logger.warning(
                    "decision %d has no search_begin_input — skipping MCTS root", i,
                )
                continue
            for k in range(K):
                tree_id = forest.add_root(
                    obs_dict, fixed_deck,
                    opp_deck_template=opp_deck,
                    iterations=getattr(config, "mcts_iterations", 64),
                    c_puct=getattr(config, "mcts_c_puct", 2.0),
                    seed=seed + i * 1000 + k,
                )
                n_attempted += 1
                if tree_id >= 0:
                    tree_map[tree_id] = (i, k)
                else:
                    n_rejected += 1
        _mp["t_add_root"] += time.perf_counter() - _t_phase
        _mp["n_roots"] += n_attempted

        # A handful of engine-refused roots per batch is normal; a wholesale
        # refusal means MCTS is silently contributing nothing, which is
        # exactly the failure mode this counter exists to make visible.
        if n_rejected:
            log = logger.error if n_rejected == n_attempted else logger.debug
            log("MCTS: %d/%d roots refused by the engine", n_rejected, n_attempted)

        if not tree_map:
            return {}

        leaf_batch = getattr(config, "mcts_leaf_batch", 512)
        while True:
            _t = time.perf_counter()
            leaves = forest.select_batch(leaf_batch)
            _mp["t_select"] += time.perf_counter() - _t
            if not leaves:
                break
            _mp["n_leaves"] += len(leaves)
            _t = time.perf_counter()
            expansions = batch_evaluate_leaves(
                leaves, policy, vocab, device,
                engine_card_features=engine_card_features,
                engine_attack_features=engine_attack_features,
            )
            _mp["t_eval"] += time.perf_counter() - _t
            _t = time.perf_counter()
            forest.expand_batch(expansions)
            _mp["t_expand"] += time.perf_counter() - _t

        results = {r["tree_id"]: r for r in forest.results()}
        _t_phase = time.perf_counter()
    finally:
        # reset, never close: close() frees the engine pool, and libcg
        # cannot give an agent's memory back.  Resetting here rather than
        # at the next shared_forest() call hands the search states back to
        # the agents' arenas now, so the next game reuses them.
        forest.reset()

    # Aggregate K determinizations
    targets: dict[int, tuple] = {}
    state_results: dict[int, list[dict]] = {}
    for tree_id, result in results.items():
        di, _k = tree_map[tree_id]
        state_results.setdefault(di, []).append(result)

    for di, k_results in state_results.items():
        agg_visits: dict[int, float] = {}
        agg_value = 0.0
        n_value = 0
        n_opts = 0
        for r in k_results:
            for opt, visits in r.get("visit_counts", []):
                agg_visits[opt] = agg_visits.get(opt, 0.0) + visits
                n_opts = max(n_opts, opt + 1)
            rv = r.get("root_value")
            if rv is not None:
                agg_value += rv
                n_value += 1
        if n_opts > 0 and agg_visits:
            total = sum(agg_visits.values())
            pi = np.zeros(n_opts, dtype=np.float32)
            for opt, v in agg_visits.items():
                if opt < n_opts and total > 0:
                    pi[opt] = float(v) / total
        else:
            pi = np.array([], dtype=np.float32)
        root_v = (agg_value / max(n_value, 1)) if n_value > 0 else None
        targets[di] = (pi, root_v)

    _mp["t_agg"] += time.perf_counter() - _t_phase
    return targets


def _eval_game_batch(
    theta: Any,
    opponent: Any,
    vocab: dict,
    fixed_deck: list[int],
    opp_deck: list[int],
    our_player: int,
    n_games: int,
    config: argparse.Namespace,
    device: torch.device,
    seed: int,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> tuple[int, int, int]:
    """Play *n_games* evaluation games.  Returns (wins, losses, draws).

    Uses RustVecEnv (no multiprocessing).  theta plays for our_player,
    opponent plays for the other side.  Both use greedy inference.

    *fixed_deck* follows theta into whichever seat it takes.  The engine
    always deals ``deck_self`` to seat 0 and ``deck_opp`` to seat 1
    irrespective of ``our_player`` (ptcg_search/src/vec_env.rs), so passing
    the decks in a fixed order would make ``our_player=1`` hand theta the
    *opponent's* archetype while the baseline plays theta's own — a deck swap
    wearing a seat swap's clothes, and the reason a stronger theta scored
    worse.  Alternating seats is meant to cancel first-player advantage, and
    only that.
    """
    from ptcg_rl.rust_vec_env import RustVecEnv
    from ptcg_rl.rollout import PolicyActor

    theta.eval()
    opponent.eval()
    actor_ours = PolicyActor(theta, vocab, device=str(device), greedy=True, seed=seed,
                              engine_card_features=engine_card_features,
                              engine_attack_features=engine_attack_features)
    actor_theirs = PolicyActor(opponent, vocab, device=str(device), greedy=True, seed=seed + 1,
                                engine_card_features=engine_card_features,
                                engine_attack_features=engine_attack_features)

    wins, losses, draws = 0, 0, 0
    n_done = 0

    deck_seat0, deck_seat1 = _decks_by_seat(fixed_deck, opp_deck, our_player)

    with RustVecEnv(
        deck_self=deck_seat0, deck_opp=deck_seat1,
        n_envs=4, our_player=our_player, seed=seed,
    ) as env:
        while n_done < n_games:
            pending = env.poll()
            if not pending:
                for t in env.drain():
                    if t.reward > 0:
                        wins += 1
                    elif t.reward < 0:
                        losses += 1
                    else:
                        draws += 1
                    n_done += 1
                if not pending:
                    time.sleep(0.001)
                    continue

            # Route: our player uses theta, theirs uses opponent
            picks_list = []
            ours_reqs = []
            theirs_reqs = []
            ours_idx = []
            theirs_idx = []
            for pi, p in enumerate(pending):
                obs = json.loads(p["obs_json"])
                if p["select_player"] == our_player:
                    ours_reqs.append({"obs": obs, "actor": "ours"})
                    ours_idx.append(pi)
                else:
                    theirs_reqs.append({"obs": obs, "actor": "theirs"})
                    theirs_idx.append(pi)

            # Build full reply list
            replies = [None] * len(pending)
            if ours_reqs:
                for ri, rep in enumerate(actor_ours(ours_reqs)):
                    replies[ours_idx[ri]] = rep
            if theirs_reqs:
                for ri, rep in enumerate(actor_theirs(theirs_reqs)):
                    replies[theirs_idx[ri]] = rep

            for rep in replies:
                if rep is not None:
                    picks_list.append(rep["picks"])
            if picks_list:
                env.reply(picks_list)

            for t in env.drain():
                if t.reward > 0:
                    wins += 1
                elif t.reward < 0:
                    losses += 1
                else:
                    draws += 1
                n_done += 1

    return wins, losses, draws


def league_evaluate(
    theta: Any,
    opponents: list[tuple[str, Any, list[int]]],
    vocab: dict,
    fixed_deck: list[int],
    config: argparse.Namespace,
    device: torch.device,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> dict[str, Any]:
    """League evaluation: θ vs every opponent, balanced seats, best decks.

    Each entry of *opponents* is ``(name, policy, deck)`` and brings its own
    decklist — see :func:`_opponent_deck`.  Both are decided by the caller, so
    there is no deck sampling here and no ``rng``: an evaluation whose opponent
    deck is drawn fresh each round is not comparable with the round before it.

    Uses RustVecEnv — no multiprocessing, no broken pipes.
    """
    import json as _json

    results: dict[str, dict] = {}
    all_passed = True

    for name, opp, opp_deck in opponents:
        # Load champion on CPU first, then move to GPU
        if isinstance(opp, str):
            opp = _load_policy(opp, Path(config.data_dir if hasattr(config, 'data_dir') else "data"),
                               torch.device("cpu"))
        opp.to(device)
        opp.eval()

        wins, losses, draws = 0, 0, 0
        total_games = 0
        # Reported per opponent because it is the difference between "θ is
        # stronger" and "θ's deck is stronger", and the two are indistinguishable
        # from the win rate alone.
        deck_label = ("mirror" if list(opp_deck) == list(fixed_deck)
                      else "own deck")

        for seat in (0, 1):
            per_seat = max(1, config.eval_games // 2)
            w, l, d = _eval_game_batch(
                theta, opp, vocab, fixed_deck, opp_deck,
                our_player=seat, n_games=per_seat,
                config=config, device=device,
                seed=config.seed + 10000 * seat,
                engine_card_features=engine_card_features,
                engine_attack_features=engine_attack_features,
            )
            wins += w
            losses += l
            draws += d
            # Count games that actually finished, not games requested: drain()
            # returns every battle that completed this cycle, so a batch can
            # overshoot `per_seat` and the requested count would divide a
            # larger numerator by a smaller denominator.
            total_games += w + l + d

        # Move opponent back to CPU to free GPU memory (except frozen_il)
        if name != "IL_baseline":
            opp.to("cpu")

        win_rate = wins / max(total_games, 1)
        passed = win_rate >= config.gate_score
        if not passed:
            all_passed = False

        results[name] = {
            "win_rate": win_rate, "wins": wins, "losses": losses,
            "draws": draws, "total": total_games, "passed": passed,
            "deck": deck_label,
        }
        logger.info("  vs %-40s  wr=%.3f  %dW/%dL/%dD  %-9s %s",
                    name[:40], win_rate, wins, losses, draws, deck_label,
                    "PASS" if passed else "FAIL")

    return {"passed": all_passed, "results": results}


def _manage_champions(
    out_dir: Path,
    policy: Any,
    current_ckpts: list[Path],
    total_games: int,
    win_rate: float,
    max_champions: int,
    deck_record: dict,
    elo: EloTracker | None = None,
) -> list[Path]:
    """Save a new champion and prune by ELO if over capacity.

    When the pool exceeds *max_champions*, the bottom half (by ELO rating)
    is dropped to bound per-round evaluation time.  Files stay on disk for
    auditing; only the active league list shrinks.
    """
    tag = f"champion-{total_games:06d}"
    _save_ckpt(out_dir, policy, tag, total_games, win_rate, deck_record=deck_record)
    new_path = out_dir / f"ckpt-mcts-{tag}.pt"
    current_ckpts.append(new_path)

    if len(current_ckpts) > max_champions:
        n_drop = min(4, len(current_ckpts) - 1)  # remove 4 lowest, keep ≥1
        # Rank by ELO (lowest first), drop n_drop
        if elo is not None:
            current_ckpts.sort(
                key=lambda p: elo.ratings.get(p.stem, elo.initial)
            )
        else:
            current_ckpts.sort(key=lambda p: p.stat().st_mtime)
        removed = current_ckpts[:n_drop]
        current_ckpts = current_ckpts[n_drop:]
        for r in removed:
            logger.info(
                "  Dropped from league (ELO %.0f): %s",
                elo.ratings.get(r.stem, 0) if elo else 0, r.name,
            )

    return current_ckpts


def _paired_corr(a: list[float], b: list[float]) -> float | None:
    """Pearson correlation, or ``None`` when it is not defined.

    Returns None rather than 0.0 for an empty or constant input: a search that
    produced no roots and a search whose values are uncorrelated with the
    outcome are opposite diagnoses, and 0.0 would report the first as the
    second.  ``None`` is dropped by the logger instead of drawing a flat line.
    """
    if len(a) < 2 or len(a) != len(b):
        return None
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _attach_actions(
    tensor_batch: dict, batch: list[dict], device: torch.device,
) -> int:
    """Overwrite the batch's action labels with the picks actually made.

    Returns the number of rows carrying at least one pick.  A row whose sample
    has no stored action keeps the empty sentinel, which every consumer already
    treats as a zero-length sequence.
    """
    width = max(
        (len(np.asarray(s["action_idx"]).ravel())
         for s in batch if s.get("action_idx") is not None),
        default=0,
    )
    if width == 0:
        return 0

    idx = np.full((len(batch), width), -1, dtype=np.int64)
    lens = np.zeros(len(batch), dtype=np.int64)
    n_with_actions = 0
    for i, s in enumerate(batch):
        a = s.get("action_idx")
        n = int(s.get("action_len", 0) or 0)
        if a is None or n <= 0:
            continue
        a = np.asarray(a, dtype=np.int64).ravel()
        idx[i, :len(a)] = a
        lens[i] = min(n, width)
        n_with_actions += 1

    tensor_batch["action_idx"] = torch.from_numpy(idx).to(device)
    tensor_batch["action_len"] = torch.from_numpy(lens).to(device)
    return n_with_actions


def train_step(
    policy: Any,
    optimizer: torch.optim.Optimizer,
    batch: list[dict],
    config: argparse.Namespace,
    device: torch.device,
    frozen_il: Any = None,
    *,
    is_first_micro: bool = True,
    is_last_micro: bool = True,
    n_micro: int = 1,
) -> dict[str, float]:
    """One training step from a sampled batch.

    The three ``*_micro`` arguments exist for gradient accumulation and
    default to a plain single step.  When accumulating, only the first
    microbatch zeroes the gradients and only the last clips and steps; the
    loss is divided by ``n_micro`` so the accumulated gradient equals the one
    the whole batch would have produced.  :func:`train_step_accum` drives
    them — callers wanting one step per batch can ignore all three.
    """
    import torch.nn.functional as F

    from ptcg_rl.search import _collate_feat_list

    if not batch:
        return {"loss": 0.0}

    policy.train()  # GRU backward needs cuDNN training mode

    feat_list = [s["features"] for s in batch]
    tensor_batch = _collate_feat_list(feat_list, device)

    h, _history_h = policy._encode(tensor_batch)
    logits, _ = policy.pointer(h, tensor_batch["tok_mask"],
                                policy.embed.card, tensor_batch)
    values = policy.value(h[:, 0])  # [B]

    # ── Actions actually taken ────────────────────────────────────────
    # `features` is featurized *before* the action is chosen, so the labels it
    # carries are the empty sentinel: action_idx all -1, action_len 0
    # (ptcg_il/featurizer.py).  recompute_logp returns 0.0 for a row with no
    # picks, so evaluating the KL on the raw feature batch compares 0 against
    # 0 for every sample — identically zero, gradient or no gradient.  The
    # real picks live on the sample, not in the features.
    _attach_actions(tensor_batch, batch, device)

    # ── KL anchor to frozen IL ────────────────────────────────────────
    kl_sum = torch.zeros((), device=device)
    kl_raw = 0.0
    n_kl = 0
    if frozen_il is not None:
        from ptcg_rl.actor import kl_to_reference, recompute_logp

        frozen_il.to(device)
        # Only the *reference* is detached.  θ's log-probs must keep their
        # graph: with the whole block under no_grad, `beta * loss_kl` adds a
        # constant to total_loss and contributes exactly zero gradient — the
        # anchor logs a number while restraining nothing.
        with torch.no_grad():
            logp_il, _ = recompute_logp(frozen_il, tensor_batch)
        logp_theta, _ = recompute_logp(policy, tensor_batch, encoded=h)

        d = kl_to_reference(logp_theta, logp_il)  # [B], mode-seeking
        # Schulman's k3, the same estimator ppo.py reports: exp(-d) - 1 + d is
        # non-negative for every d, so minimising it is a penalty.  The raw
        # mean of d is not — its minimiser drives log π_θ(a) to -∞, which as a
        # loss term is an instruction to abandon the actions θ just took.
        # The clamp is overflow hygiene only (exp(88) is inf in fp32), far
        # outside the range a live anchor occupies.
        d_safe = d.clamp(-20.0, 20.0)
        kl_sum = ((-d_safe).exp() - 1.0 + d_safe).sum()
        kl_raw = float(d.detach().mean())
        n_kl = d.shape[0]

    total_loss = torch.zeros((), device=device)
    loss_pi = torch.zeros((), device=device)
    loss_v = torch.zeros((), device=device)
    n_pi = 0

    # ── Advantage weights for the behaviour-cloning term ──────────────
    # Cloning θ's own actions unweighted reinforces losing lines exactly as
    # hard as winning ones: it is not an RL objective at all, just a mode
    # sharpener.  Weighting by return − V(s) turns it into REINFORCE with the
    # critic as baseline, so a losing action gets pushed *down*.  Normalised
    # per minibatch, matching rollout.normalize_advantages (§9.1) — advantages
    # are never normalised twice, and this is the only place it happens here.
    from ptcg_rl.rollout import normalize_advantages

    # ── Critic targets: the game outcome, per AlphaZero ───────────────
    # l = (z - v)² - π_MCTS·log p: the *policy* target comes from search, the
    # *value* target is the final result z.  Training v on the search root
    # value instead closes a loop with no external signal in it — the leaves
    # of a 64-simulation search are this same critic, so the target is a
    # smoothed copy of the prediction and any constant is a fixed point.
    # Measured once: target std collapsed 1.0 → 0.022, v_mse 0.0798 → 0.0006,
    # ev 0.669 → -0.199.  The MSE fell because predicting a constant is easy.
    v_targets = [float(s["game_outcome"]) for s in batch]

    # `mcts_value` is kept as a *diagnostic* against the same decision's
    # outcome.  Near-zero correlation means the search sees nothing the
    # outcome agrees with, which condemns `mcts_pi` too — the policy would be
    # distilling noise, and that is invisible in pi_ce, which only measures
    # agreement with whatever the visit counts happened to be.
    search_v = [s.get("mcts_value") for s in batch]
    n_search_targets = sum(1 for v in search_v if v is not None)
    search_outcome_corr = _paired_corr(
        [float(v) for v in search_v if v is not None],
        [float(s["game_outcome"]) for s, v in zip(batch, search_v)
         if v is not None],
    )

    rets_t = torch.tensor(v_targets, device=device, dtype=torch.float32)
    adv = normalize_advantages(rets_t - values.detach().float())

    # The critic term is elementwise, so it is one vectorised MSE rather than
    # `len(batch)` scalar ones summed in Python.  `rets_t` is the same vector
    # of targets built just above for the advantage.
    loss_v = F.mse_loss(values, rets_t.to(values.dtype), reduction="sum")
    rets: list[float] = list(v_targets)
    preds: list[float] = values.detach().float().tolist()

    # Split the batch by which policy term it takes, then run each term once
    # over its rows.  The behaviour-cloning branch used to call
    # `recompute_logp` per sample on a batch of one, and every one of those
    # calls retained its own autograd graph through the AR pointer loop — up
    # to `len(batch)` live graphs where one suffices.  Activations are 99% of
    # this step's memory (the model is 6.1M params, 0.12 GB with grads and
    # Adam states), so that was the difference between fitting in 16 GiB and
    # not.  Identical loss and gradients either way; `actor.py`'s AR loop
    # already had the same fix applied one level down.
    bc_rows: list[int] = []
    for i, sample in enumerate(batch):
        mcts_pi = sample.get("mcts_pi")
        if mcts_pi is not None and len(mcts_pi) > 0:
            # MCTS distillation: CE to visit distribution π̃.  Kept per-row:
            # π̃ is ragged (one entry per legal option at *that* state), so
            # batching it would need a pad-and-mask that costs more than the
            # loop saves — and unlike the BC branch it builds no extra graph,
            # reading the already-computed full-batch `logits`.
            pi_tilde = torch.as_tensor(
                np.asarray(mcts_pi, dtype=np.float32)
            ).to(device)
            mask = tensor_batch["opt_mask"][i].bool()
            masked = logits[i].float().masked_fill(~mask, float("-inf"))
            logp = F.log_softmax(masked, dim=-1)
            n_legal = min(len(pi_tilde), int(mask.sum()))
            if n_legal > 0:
                loss_pi = loss_pi - (pi_tilde[:n_legal] * logp[:n_legal]).sum()
                n_pi += 1
        elif (sample.get("action_idx") is not None
                and sample.get("action_len", 0) > 0):
            bc_rows.append(i)

    if bc_rows:
        # Behaviour cloning: maximise the joint log-prob of the full AR
        # sequence (including STOP).  `_attach_actions` already put the real
        # picks on `tensor_batch`, so the rows are just a gather.
        from ptcg_rl.actor import recompute_logp

        rows = torch.tensor(bc_rows, device=device, dtype=torch.long)
        x_bc = {k: v[rows] for k, v in tensor_batch.items()}
        logp_bc, _ = recompute_logp(policy, x_bc, encoded=h[rows])
        # A negative advantage flips the sign: the action becomes something to
        # move probability away from, not toward.
        loss_pi = loss_pi - (adv[rows] * logp_bc).sum()
        n_pi += len(bc_rows)

    n_batch = len(batch)
    loss_pi = loss_pi / max(n_pi, 1)
    loss_v = loss_v / n_batch
    loss_kl = kl_sum / max(n_kl, 1)
    total_loss = config.c_pi * loss_pi + config.c_value * loss_v \
                 + getattr(config, "beta", 0.1) * loss_kl

    if is_first_micro:
        optimizer.zero_grad(set_to_none=True)
    (total_loss / n_micro).backward()

    # Pre-clip gradient norm.  Only meaningful once every microbatch has
    # contributed, so it is measured — and reported — on the last one.
    raw_grad = 0.0
    if is_last_micro:
        for p in policy.parameters():
            if p.grad is not None:
                raw_grad += p.grad.norm().item() ** 2
        raw_grad = raw_grad ** 0.5

        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
        optimizer.step()

    # Explained variance: 1 - Var(ret - pred) / Var(ret)
    rets_arr = np.array(rets, dtype=np.float32)
    var_ret = float(np.var(rets_arr))
    if var_ret > 1e-8:
        ev = 1.0 - float(np.var(rets_arr - np.array(preds, dtype=np.float32))) / var_ret
    else:
        ev = float("nan")

    return {
        "loss": float(total_loss.detach()),
        "pi_ce": float(loss_pi.detach()) if n_pi > 0 else 0.0,
        "v_mse": float(loss_v.detach()),
        # `kl` stays the raw mean of log π_θ(a) − log π_IL(a), so it is
        # comparable with ppo.py's `kl_to_il`; `kl_k3` is the non-negative
        # quantity actually added to the loss.
        "kl": kl_raw,
        "kl_k3": float(loss_kl.detach()),
        "n_pi": n_pi,
        "search_target_frac": n_search_targets / max(len(batch), 1),
        "grad_norm": raw_grad,
        "ev": ev,
        "pred_mean": float(np.mean(preds)),
        "pred_std": float(np.std(preds)),
        "ret_mean": float(np.mean(rets)),
        # |adv| collapsing toward 0 means the BC term has stopped pushing in
        # either direction — the objective has gone quiet, which unweighted
        # cloning could never reveal because its weight was always 1.
        "adv_abs_mean": float(adv.abs().mean()),
        # Spread of the critic target.  ev is a *ratio* against this, so it
        # goes negative on a near-constant target however good the fit is —
        # read the two together or ev will look like a regression when only
        # the target changed.
        "target_std": float(np.std(rets_arr)),
        "search_outcome_corr": search_outcome_corr,
    }


#: Stats that are counts over the batch and must be summed across
#: microbatches; everything else is a per-row mean and is averaged.
_ACCUM_SUM_KEYS = frozenset({"n_pi"})


def train_step_accum(
    policy: Any,
    optimizer: torch.optim.Optimizer,
    batch: list[dict],
    config: argparse.Namespace,
    device: torch.device,
    frozen_il: Any = None,
    grad_accum: int = 1,
) -> dict[str, float]:
    """:func:`train_step` over *grad_accum* microbatches, one optimizer step.

    The effective batch is unchanged — the gradient is the same average the
    whole batch would have produced — but only ``len(batch) / grad_accum``
    rows' activations are alive at a time, and activations are ~99% of this
    step's memory.

    Not bit-identical to ``grad_accum=1``: ``normalize_advantages`` runs per
    microbatch, so advantages are standardised over the microbatch rather than
    the full batch.  That is a change in the advantage *scale*, not in which
    actions are reinforced, and it shrinks as the microbatch grows.  It is the
    one reason to leave this at 1 when memory allows.
    """
    if grad_accum <= 1 or len(batch) <= 1:
        return train_step(policy, optimizer, batch, config, device, frozen_il)

    n_micro = min(grad_accum, len(batch))
    # Contiguous near-equal chunks.  The batch is already a uniform random
    # sample, so any split is as unbiased as any other.
    bounds = [round(k * len(batch) / n_micro) for k in range(n_micro + 1)]
    chunks = [batch[bounds[k]:bounds[k + 1]] for k in range(n_micro)]
    chunks = [c for c in chunks if c]
    n_micro = len(chunks)

    merged: dict[str, float] = {}
    weights = 0
    for k, chunk in enumerate(chunks):
        stats = train_step(
            policy, optimizer, chunk, config, device, frozen_il,
            is_first_micro=(k == 0),
            is_last_micro=(k == n_micro - 1),
            n_micro=n_micro,
        )
        w = len(chunk)
        weights += w
        for key, val in stats.items():
            if val is None:
                # Keep the key. `search_outcome_corr` is None when search
                # produced nothing to correlate, and "absent" reads as a
                # different diagnosis downstream than "present and None".
                merged.setdefault(key, None)
                continue
            if merged.get(key, 0.0) is None:
                # A later microbatch had data where an earlier one did not.
                merged[key] = 0.0
            if key in _ACCUM_SUM_KEYS:
                merged[key] = merged.get(key, 0.0) + val
            elif key == "grad_norm":
                # Measured only on the last microbatch, where it is the norm
                # of the *whole* accumulated gradient — averaging it with the
                # zeros the others report would understate it by n_micro.
                merged[key] = max(merged.get(key, 0.0), val)
            elif isinstance(val, (int, float)):
                # A non-finite microbatch poisons the mean, and that is the
                # correct outcome: callers read `loss`/`ev` unconditionally,
                # so dropping the key would raise KeyError, and silently
                # skipping the value would hide a NaN the run needs to see.
                prev = merged.get(key, 0.0)
                merged[key] = (float("nan") if not math.isfinite(float(val))
                               else prev + float(val) * w)
                if isinstance(prev, float) and math.isnan(prev):
                    merged[key] = float("nan")
            else:
                merged[key] = val

    for key in list(merged):
        if key in _ACCUM_SUM_KEYS or key == "grad_norm":
            continue
        if isinstance(merged[key], float) and weights:
            merged[key] /= weights
    return merged



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load artifacts ──────────────────────────────────────────────────
    logger.info("Loading artifacts from %s", data_dir)
    vocab = _load_vocab(data_dir)
    archetypes = _load_archetypes(data_dir)
    _check_ckpt_deck(args.il_ckpt, args.deck_archetype)
    # θ_init is checked too when it is a different file: it is the model that
    # actually plays this deck, so a mismatch there is the one that feeds
    # all-zero card features.
    if args.init_ckpt and args.init_ckpt != args.il_ckpt:
        _check_ckpt_deck(args.init_ckpt, args.deck_archetype)
    fixed_deck = _deck_for(data_dir, args.deck_archetype)
    logger.info(
        "Deck: %s (%d cards)",
        "fixed_deck" if args.deck_archetype is None
        else f"archetype {args.deck_archetype}",
        len(fixed_deck),
    )
    opp_decks = _load_opp_decks(data_dir, all_archetypes=args.all_archetypes)
    # Load engine card/attack feature maps (required for pure-feature model)
    import numpy as _np
    engine_card_features = None
    engine_attack_features = None
    ecf_path = data_dir / "engine_card_features.npy"
    eaf_path = data_dir / "engine_attack_features.npy"
    if ecf_path.exists():
        engine_card_features = _np.load(ecf_path, allow_pickle=True).item()
        logger.info("Engine card features loaded (%d cards)", len(engine_card_features))
    if eaf_path.exists():
        engine_attack_features = _np.load(eaf_path, allow_pickle=True).item()
        logger.info("Engine attack features loaded (%d attacks)", len(engine_attack_features))
    opp_weights: dict[int, float] = {}  # adaptive: low wr → high weight

    logger.info(
        "Opponent archetypes: %s",
        ", ".join(f"id={d['id']}" for d in opp_decks),
    )

    # ── ELO tracker (must be before policy load — auto-resume needs it) ──
    elo = EloTracker()
    # Discover all IL checkpoints and register them as fixed-reference
    # opponents.  ckpt-best is the anchor at 1500 (fixed).
    il_checkpoints = _discover_il_checkpoints(args.il_ckpt)
    logger.info(
        "IL checkpoints in %s: %d found",
        Path(args.il_ckpt).resolve().parent,
        len(il_checkpoints),
    )
    for name, _path in il_checkpoints:
        elo.add_model(name, fixed=(name == "ckpt-best"))
    elo.add_model("IL_baseline")  # frozen π_IL (may duplicate a step ckpt)
    # Restore saved ELO ratings — must come after add_model so fixed
    # markers are re-applied, but overwrites fresh 1500s with saved values.
    _load_elo(out_dir, elo)
    elo_name = lambda g: f"ckpt-mcts-champion-{g:06d}"

    # ── Discover existing champions ─────────────────────────────────────
    #
    # Gated champions from *this* out_dir, and nothing else.  Two exclusions,
    # both deliberate:
    #
    #   • the --il-ckpt directory.  Its checkpoints are already league
    #     opponents via `il_checkpoints` below, so folding them in here entered
    #     each of them twice — double the eval games, double the ELO updates.
    #     Worse for resume: `ckpt-best` carries a *fixed* 1500, so any branch
    #     whose champions sat below that would "auto-resume from best champion"
    #     into the IL model and silently discard its own MCTS progress.
    #   • `ckpt-mcts-last.pt`.  It is written unconditionally at the end of a
    #     run, gate or no gate, so it is not a result and must not be resumed
    #     from or played against.
    #
    # sorted() because the ordering used to decide the outcome: every ELO
    # lookup missed (see `elo_key`), so `_find_best_champion` ranked a flat
    # field and returned whatever glob yielded first.
    champion_ckpts: list[Path] = sorted(
        out_dir.glob("ckpt-mcts-champion-*.pt"))
    # ── Load policy ─────────────────────────────────────────────────────
    # Resolve what checkpoint to start from:
    #   1. --resume (explicit) → use that file
    #   2. highest-ELO champion in out_dir → auto-resume
    #   3. fall back → --init-ckpt if given, else --il-ckpt
    #
    # θ_init and the anchor are separate inputs: pre-training on every
    # archetype and fine-tuning on one means starting from the generalist
    # while still being gated against that deck's specialist.
    init_ckpt = args.init_ckpt or args.il_ckpt
    best_ckpt: Path | None = None
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        policy = _load_policy(args.resume, data_dir, device)
    else:

        best_ckpt = _find_best_champion(champion_ckpts, elo)
        if best_ckpt is not None:
            best_elo_val = elo.ratings.get(elo_key(best_ckpt), 0.0)
            logger.info(
                "Auto-resuming from best champion: %s (ELO: %.0f)",
                best_ckpt.name, best_elo_val,
            )
            policy = _load_policy(str(best_ckpt), data_dir, device)
        else:
            logger.info(
                "Loading warm-start from %s%s", init_ckpt,
                " (--init-ckpt; anchor stays "
                f"{args.il_ckpt})" if args.init_ckpt
                and args.init_ckpt != args.il_ckpt else "",
            )
            policy = _load_policy(init_ckpt, data_dir, device)

    frozen_il = _load_frozen_anchor(args.il_ckpt, data_dir, device)
    logger.info("Frozen π_IL loaded for evaluation")

    # ── Seat-1 pilots ───────────────────────────────────────────────────
    # θ is a specialist on `fixed_deck`; seat 1 is dealt a *sampled* archetype.
    # With θ answering both seats it plays that deck with cards it has never
    # seen, so beating it measures nothing.  Registering the model that owns
    # each deck — and the generalist for the rest of the 179-deck pool — makes
    # the opponent competent without needing a league.
    opp_experts = OpponentExperts(
        default_ckpt=args.opp_default_ckpt,
        experts=parse_opp_experts(args.opp_expert),
        data_dir=data_dir,
        device=device,
        vocab=vocab,
        engine_card_features=engine_card_features,
        engine_attack_features=engine_attack_features,
    )
    if not 0.0 <= args.opp_expert_share <= 1.0:
        raise SystemExit(
            f"--opp-expert-share {args.opp_expert_share} is not a fraction "
            "in [0, 1]"
        )
    logger.info("Seat-1 pilots : %s", opp_experts.describe())
    if opp_experts.experts:
        logger.info(
            "Rival share   : %.0f%% of games drawn from %d expert deck(s); "
            "the rest from the full %d-deck pool",
            args.opp_expert_share * 100, len(opp_experts.experts),
            len(opp_decks),
        )
    if not opp_experts.enabled:
        logger.warning(
            "No --opp-default-ckpt or --opp-expert — θ pilots the opponent "
            "seat as well, and sp/win_rate will read high for that reason "
            "alone",
        )

    # ── Deck label for the champions written here (Kaggle submission) ────
    #
    # Built from --deck-archetype, not copied from the IL checkpoint.  What a
    # champion plays is `fixed_deck` above, which `_deck_for` derives from
    # --deck-archetype; the IL record describes what that *checkpoint* was
    # trained on, and the two differ by construction whenever θ_init is a
    # generalist.  A generalist's record carries `deck = archetypes.json's
    # fixed_deck`, which in this corpus is archetype 21's list — so copying it
    # onto a run playing archetype 1 shipped a champion labelled with a deck
    # it never plays, and nothing downstream would have caught it.
    #
    # Fatal, and fatal *here* rather than at the first promotion.  This used to
    # swallow every exception and continue with `_deck_record = None`, which
    # produced champions with no deck label after hours of self-play — and an
    # unlabelled champion is exactly what the league now drops and what
    # build_submission refuses to ship.  Nothing has been computed yet at this
    # point, so refusing costs nothing.
    from ptcg_il.deck import require_deck_record

    try:
        _deck_record = require_deck_record(
            _deck_record_for(data_dir, args.deck_archetype),
            f"--deck-archetype {args.deck_archetype}",
        )
    except Exception as e:  # noqa: BLE001 — surfaced as a clean exit, not a trace
        raise SystemExit(
            f"Cannot build a deck record for --deck-archetype "
            f"{args.deck_archetype}: {e}\nRefusing to start — every champion "
            f"this run writes would be unlabelled."
        ) from e
    logger.info(
        "Deck label: %s (%d cards) — built from --deck-archetype",
        "fixed_deck" if args.deck_archetype is None
        else f"archetype {args.deck_archetype}",
        _deck_record.get("deck_size", 0),
    )

    # ── Startup diagnostic ──────────────────────────────────────────────
    logger.info("══════════════════════════════════════════════════════════")
    logger.info("IL checkpoint : %s", args.il_ckpt)
    logger.info("Output dir    : %s", out_dir)
    logger.info("Champions     : %d found (max %d)", len(champion_ckpts), args.max_champions)
    if champion_ckpts:
        for ckpt in sorted(champion_ckpts, key=lambda p: p.stat().st_mtime):
            r = elo.ratings.get(elo_key(ckpt))
            logger.info("  %-45s  ELO=%s", ckpt.name,
                        f"{r:.0f}" if r is not None else "?")
    else:
        logger.info("  (none — will start from IL warm-start)")
    logger.info("Policy source : %s",
                "explicit --resume" if args.resume
                else f"auto-resume {best_ckpt.name}" if best_ckpt is not None
                else f"warm-start {init_ckpt}")
    logger.info("Frozen KL anchor : %s", args.il_ckpt)
    logger.info("ELO leaderboard:")
    for line in elo.leaderboard().split("\n"):
        logger.info("  %s", line)
    logger.info("══════════════════════════════════════════════════════════")

    # ── W&B logger ────────────────────────────────────────────────────
    wb_cfg = {k: str(v) for k, v in vars(args).items()}
    wb_cfg.update({"n_opp_archetypes": len(opp_decks), "il_ckpt": args.il_ckpt})
    wb = _MctsWandbLogger(args, wb_cfg)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    # ── Replay buffer ───────────────────────────────────────────────────
    from ptcg_rl.search import ReplayBuffer
    replay = ReplayBuffer(capacity=args.buffer_capacity)

    # ── League state (champion_ckpts already discovered above) ─────────

    # ── Training loop ───────────────────────────────────────────────────
    total_games = 0
    t_start = time.perf_counter()
    t_spent_sp = 0.0     # cumulative self-play time
    t_spent_train = 0.0  # cumulative training time
    t_spent_eval = 0.0   # cumulative evaluation time
    cum_train_loss = 0.0
    cum_grad_norm = 0.0
    n_train_steps = 0
    iter_count = 0

    def _print_summary():
        elapsed = time.perf_counter() - t_start
        logger.info(
            "ITER %-3d │ games %d │ wall %s (sp %s + train %s + eval %s) │ "
            "buf %d │ avg_loss %.4f │ avg_grad %.3f",
            iter_count, total_games,
            fmt_dur(elapsed), fmt_dur(t_spent_sp), fmt_dur(t_spent_train),
            fmt_dur(t_spent_eval),
            len(replay), cum_train_loss, cum_grad_norm,
        )

    while total_games < args.total_games:
        iter_count += 1
        n_games = min(args.games_per_iter, args.total_games - total_games)
        logger.info("━━━ iter %d  self-play %d games (total %d/%d) ━━━",
                    iter_count, n_games, total_games + n_games, args.total_games)

        # ── Phase 1: Self-play ────────────────────────────────────────
        t_phase = time.perf_counter()
        samples, sp_stats = run_self_play_games(
            policy, vocab, archetypes, fixed_deck, opp_decks,
            args, device, rng, n_games,
            opp_weights=opp_weights,
            engine_card_features=engine_card_features,
            engine_attack_features=engine_attack_features,
            opp_experts=opp_experts,
        )
        t_sp = time.perf_counter() - t_phase
        t_spent_sp += t_sp
        wb.log_sp(sp_stats, total_games)
        for s in samples:
            replay.push(
                features=s["features"],
                action_idx=s["action_idx"],
                action_len=s["action_len"],
                mcts_pi=s["mcts_pi"],
                mcts_value=s["mcts_value"],
                game_outcome=s["game_outcome"],
                opp_archetype=s["opp_archetype"],
            )
        del samples  # free transient list (features now owned by replay buffer)
        total_games += n_games
        # Print self-play summary
        mcts_pct = sp_stats["sp_mcts_decisions"] / max(sp_stats["sp_decisions"], 1) * 100
        logger.info(
            "SP  │ %d games in %s (%.1f/min) │ wr=%.2f %dW/%dL/%dD │ "
            "%d dec (%d MCTS, %.0f%%) │ %d train / %d opp-seat │ "
            "piloted %.0f%% (expert %.0f%%)",
            n_games, fmt_dur(t_sp), sp_stats["sp_games_per_min"],
            sp_stats["sp_win_rate"], sp_stats["sp_wins"],
            sp_stats["sp_losses"], sp_stats["sp_draws"],
            sp_stats["sp_decisions"], sp_stats["sp_mcts_decisions"], mcts_pct,
            sp_stats["sp_samples"], sp_stats["sp_dropped_seat"],
            sp_stats["sp_opp_piloted_frac"] * 100,
            sp_stats["sp_expert_opp_frac"] * 100,
        )

        # ── Phase 2: Train ────────────────────────────────────────────
        if len(replay) >= args.min_buffer:
            policy.train()
            logger.info(
                "TRAIN │ %d steps  batch=%d%s  lr=%.1e",
                args.train_steps_per_iter, args.batch_size,
                (f" (×{args.grad_accum} accum → "
                 f"{max(1, args.batch_size // args.grad_accum)}/micro)")
                if args.grad_accum > 1 else "",
                args.lr,
            )
            t_phase = time.perf_counter()
            cum_pi = 0.0
            cum_v = 0.0
            cum_g = 0.0
            cum_ev = 0.0
            n_ev = 0            # ev is NaN when the return batch is constant
            cum_kl = 0.0
            cum_n_pi = 0
            cum_pred_mean = 0.0
            cum_pred_std = 0.0
            cum_ret_mean = 0.0
            cum_adv = 0.0
            cum_kl_k3 = 0.0
            cum_search_tgt = 0.0
            cum_target_std = 0.0
            cum_soc = 0.0
            n_soc = 0      # steps where the correlation was defined at all
            cum_clipped = 0      # steps whose raw grad norm exceeded the clip
            cum_loss = 0.0       # the objective actually optimized
            total_lr_steps = (args.total_games // args.games_per_iter) * args.train_steps_per_iter
            for step in range(args.train_steps_per_iter):
                if args.lr_schedule == "cosine":
                    lr = _cosine_lr(n_train_steps, total_lr_steps,
                                    args.lr, args.min_lr, args.warmup_steps)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                batch = replay.sample(args.batch_size, rng)
                stats = train_step_accum(
                    policy, optimizer, batch, args, device,
                    frozen_il=frozen_il, grad_accum=args.grad_accum,
                )
                cum_pi += stats["pi_ce"]
                cum_v += stats["v_mse"]
                cum_g += stats["grad_norm"]
                ev_step = stats.get("ev", float("nan"))
                if math.isfinite(ev_step):
                    cum_ev += ev_step
                    n_ev += 1
                cum_kl += stats.get("kl", 0.0)
                cum_n_pi += stats.get("n_pi", 0)
                cum_pred_mean += stats.get("pred_mean", 0.0)
                cum_pred_std += stats.get("pred_std", 0.0)
                cum_ret_mean += stats.get("ret_mean", 0.0)
                cum_adv += stats.get("adv_abs_mean", 0.0)
                cum_kl_k3 += stats.get("kl_k3", 0.0)
                cum_search_tgt += stats.get("search_target_frac", 0.0)
                cum_target_std += stats.get("target_std", 0.0)
                soc = stats.get("search_outcome_corr")
                if soc is not None and math.isfinite(soc):
                    cum_soc += soc
                    n_soc += 1
                cum_clipped += int(stats["grad_norm"] > args.grad_clip)
                cum_loss += stats["loss"]
                n_train_steps += 1
                if (step + 1) % 100 == 0:
                    logger.info(
                        "  step %d/%d  lr=%.1e  loss=%.4f  pi=%.4f  v=%.4f  "
                        "kl=%.4f  ev=%.3f  |g|=%.2f→%.3f",
                        step + 1, args.train_steps_per_iter,
                        optimizer.param_groups[0]["lr"],
                        stats["loss"], stats["pi_ce"], stats["v_mse"],
                        stats.get("kl", 0),
                        stats.get("ev", float("nan")),
                        stats["grad_norm"],
                        min(stats["grad_norm"], args.grad_clip),                     )
            t_tr = time.perf_counter() - t_phase
            t_spent_train += t_tr
            n_steps = args.train_steps_per_iter
            avg_pi = cum_pi / n_steps
            avg_v = cum_v / n_steps
            avg_g = cum_g / n_steps
            avg_ev = cum_ev / n_ev if n_ev else float("nan")
            avg_kl = cum_kl / n_steps
            avg_target_std = cum_target_std / n_steps
            cum_train_loss = (cum_train_loss * (n_train_steps - n_steps)
                              + (avg_pi + args.c_value * avg_v) * n_steps
                             ) / max(n_train_steps, 1)
            cum_grad_norm = (cum_grad_norm * (n_train_steps - n_steps)
                             + avg_g * n_steps) / max(n_train_steps, 1)
            logger.info(
                "TRAIN │ %d steps in %s │ pi_ce=%.4f  v_mse=%.4f  kl=%.4f  "
                "ev=%.3f (tgt_std=%.3f)  grad=%.3f (clipped %.0f%%)  mcts~z=%s",
                n_steps, fmt_dur(t_tr), avg_pi, avg_v,
                avg_kl, avg_ev, avg_target_std, avg_g,
                cum_clipped / n_steps * 100,
                f"{cum_soc / n_soc:+.3f}" if n_soc else "n/a",
            )
            wb.log_train({"pi_ce": avg_pi, "v_mse": avg_v,
                          # ev is NaN on a constant-return batch; log_train
                          # drops None rather than poisoning the series.
                          "ev": avg_ev if math.isfinite(avg_ev) else None,
                          "kl": avg_kl,
                          "grad": avg_g,
                          "grad_clip_frac": cum_clipped / n_steps,
                          # train_step's own total (c_pi·π + c_value·v + β·kl);
                          # re-deriving it here dropped c_pi and β silently.
                          "loss": cum_loss / n_steps,
                          "lr": optimizer.param_groups[0]["lr"],
                          # Critic-collapse watch: pred_std → 0 against a
                          # non-zero ret_mean spread is the R1 failure mode.
                          "pred_mean": cum_pred_mean / n_steps,
                          "pred_std": cum_pred_std / n_steps,
                          "ret_mean": cum_ret_mean / n_steps,
                          "adv_abs_mean": cum_adv / n_steps,
                          "kl_k3": cum_kl_k3 / n_steps,
                          "beta": args.beta,
                          # Share of the batch whose critic target came from
                          # search rather than the game outcome.  0 with
                          # --mcts-distill on means the roots are being
                          # refused, not that search is agreeing.
                          "search_target_frac": cum_search_tgt / n_steps,
                          "target_std": avg_target_std,
                          # None when search produced nothing to correlate —
                          # log_train drops it rather than drawing a 0.0 that
                          # would read as "search is uninformative".
                          "search_outcome_corr": (
                              cum_soc / n_soc if n_soc else None),
                          "n_pi": cum_n_pi / n_steps,
                          "buffer": len(replay),
                          "sec": t_tr, "steps": n_steps},
                         total_games)
            if total_games % 500 < args.games_per_iter:
                wb.log_model(policy, total_games)
        else:
            logger.info(
                "TRAIN │ skipped (buffer %d < min %d)",
                len(replay), args.min_buffer,
            )

        # ── Phase 3: League evaluation ────────────────────────────────
        if total_games % args.eval_every_games < args.games_per_iter:
            # Every opponent brings the deck its own checkpoint names; one
            # without a `deck` record is dropped rather than handed a guess.
            opponents: list[tuple[str, Any, list[int]]] = []

            def _enlist(name: str, model: Any, record: dict | None) -> None:
                deck = _opponent_deck(record, fixed_deck)
                if deck is None:
                    logger.warning(
                        "  Skipping %s: no 'deck' record, cannot pick a deck "
                        "for it", name,
                    )
                    return
                opponents.append((name, model, deck))

            _enlist("IL_baseline", frozen_il, _deck_record_of(args.il_ckpt))
            # Load IL checkpoints (skip the one that's already frozen_il)
            il_ckpt_resolved = str(Path(args.il_ckpt).resolve())
            for ckpt_name, ckpt_path in il_checkpoints:
                if str(Path(ckpt_path).resolve()) == il_ckpt_resolved:
                    continue  # already loaded as frozen_il / IL_baseline
                try:
                    ckpt_model, rec = _load_policy_and_deck(
                        ckpt_path, data_dir, device)
                    _enlist(ckpt_name, ckpt_model, rec)
                except Exception as e:
                    logger.warning("  Skipping unloadable IL ckpt %s: %s",
                                   ckpt_name, e)
            for ckpt_path in sorted(champion_ckpts, key=lambda p: p.stat().st_mtime):
                try:
                    champ, rec = _load_policy_and_deck(
                        str(ckpt_path), data_dir, device)
                    _enlist(ckpt_path.stem, champ, rec)
                except Exception as e:
                    logger.warning("  Skipping unloadable champion %s: %s",
                                   ckpt_path.name, e)

            if not opponents:
                # `all_passed` starts True, so an empty league would report
                # PASS on no games at all and promote a champion on it.
                logger.error(
                    "EVAL │ no opponent carries a deck record — skipping this "
                    "round rather than passing a gate vacuously",
                )
                _print_summary()
                continue

            logger.info(
                "EVAL │ %d opponents × %d games  gate ≥%.0f%%",
                len(opponents), args.eval_games, args.gate_score * 100,
            )
            t_phase = time.perf_counter()
            league_result = league_evaluate(
                policy, opponents, vocab, fixed_deck,
                args, device,
                engine_card_features=engine_card_features,
                engine_attack_features=engine_attack_features,
            )
            t_ev = time.perf_counter() - t_phase
            t_spent_eval += t_ev

            all_wr = [r["win_rate"] for r in league_result["results"].values()]
            avg_wr = sum(all_wr) / max(len(all_wr), 1)
            logger.info(
                "EVAL │ %d opp in %s │ avg=%.3f min=%.3f max=%.3f",
                len(opponents), fmt_dur(t_ev), avg_wr, min(all_wr), max(all_wr),
            )
            for name, r in league_result["results"].items():
                logger.info(
                    "  %-40s  wr=%.3f  %s",
                    name[:40], r["win_rate"],
                    "PASS" if r["passed"] else "FAIL",
                )

            wb.log_eval(league_result, len(champion_ckpts), total_games)

            # ── ELO update ──────────────────────────────────────────────
            new_name = elo_name(total_games)
            elo.add_model(new_name)
            for name, r in league_result["results"].items():
                elo.update(new_name, name, r["wins"], r["losses"])
            new_elo = elo.ratings[new_name]
            threshold = elo.top_pct(0.70)
            best_name, best_elo = elo.best()

            logger.info("ELO │ %s: %.0f | top30%%=%.0f | best=%s(%.0f)",
                        new_name, new_elo, threshold, best_name, best_elo)
            logger.info("ELO leaderboard:\n%s", elo.leaderboard())
            wb.log_elo(elo, new_name, threshold, total_games)

            # Persist ELO ratings
            _save_elo(out_dir, elo)

            if new_elo >= threshold:
                logger.info("ELO │ PASSED (%.0f ≥ top30%% %.0f) — saving", new_elo, threshold)
                champion_ckpts = _manage_champions(
                    out_dir, policy, champion_ckpts, total_games,
                    new_elo / 1000.0, args.max_champions, elo=elo,
                    deck_record=_deck_record,
                )
            else:
                logger.info("ELO │ FAILED (%.0f < top30%% %.0f)", new_elo, threshold)
        else:
            # Per-iteration summary when no eval
            _print_summary()

    # ── Final ──────────────────────────────────────────────────────────
    _save_ckpt(out_dir, policy, "last", total_games, None, deck_record=_deck_record)
    elapsed = time.perf_counter() - t_start
    logger.info(
        "DONE │ %d games in %s │ sp=%s train=%s eval=%s │ %d champions",
        total_games, fmt_dur(elapsed),
        fmt_dur(t_spent_sp), fmt_dur(t_spent_train), fmt_dur(t_spent_eval),
        len(champion_ckpts),
    )
    wb.log_model(policy, total_games)
    _save_elo(out_dir, elo)
    wb.finish()
    return 0


def _wb_key(name: Any) -> str:
    """Sanitize a run-time name into one W&B panel segment.

    Opponent and ELO keys come from checkpoint stems, which may contain ``/``
    — that would silently split the metric into another nesting level and
    scatter one series across two panel groups.
    """
    return str(name).replace("/", "_")


class _MctsWandbLogger:
    """Lightweight W&B logger for MCTS self-play training."""

    def __init__(self, args: argparse.Namespace, cfg: dict):
        self.active = False
        if args.no_wandb or not args.wandb:
            return
        try:
            import wandb
        except ImportError:
            return
        name = args.wandb_name or f"mcts-{Path(args.il_ckpt).parent.name}"
        try:
            self._run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=name,
                config=cfg,
                mode=args.wandb_mode,
                resume="allow",
            )
            self._run.define_metric("mcts/step")
            for prefix in ("mcts", "sp", "train", "eval", "elo", "model", "grad"):
                self._run.define_metric(f"{prefix}/*", step_metric="mcts/step")
            self.active = True
        except Exception as e:
            logger.warning("wandb.init failed: %s", e)

    def log(self, metrics: dict, step: int | None = None) -> None:
        if not self.active:
            return
        payload = {k: v for k, v in metrics.items() if v is not None}
        if step is not None:
            payload["mcts/step"] = step
        self._run.log(payload)

    def log_sp(self, stats: dict, step: int) -> None:
        payload = {
            "sp/win_rate": stats["sp_win_rate"],
            "sp/games": stats["sp_games"],
            "sp/wins": stats["sp_wins"],
            "sp/losses": stats["sp_losses"],
            "sp/draws": stats["sp_draws"],
            "sp/sec": stats["sp_sec"],
            "sp/games_per_min": stats["sp_games_per_min"],
            "sp/dec_per_game": stats["sp_dec_per_game"],
            "sp/mcts_fraction": stats["sp_mcts_decisions"]
            / max(stats["sp_decisions"], 1),
            "sp/decisions": stats["sp_decisions"],
            "sp/mcts_decisions": stats["sp_mcts_decisions"],
            # Decisions θ played on the opponent seat and did not learn from.
            # Without a seat-1 pilot this is roughly half of sp/decisions;
            # with one it is 0, because the opponent's decisions are never
            # recorded in the first place.  Anything between the two means the
            # seat routing has changed under us.
            "sp/samples": stats.get("sp_samples"),
            "sp/dropped_seat": stats.get("sp_dropped_seat"),
            # Read the win rate against this.  At 0.0 seat 1 is θ playing an
            # archetype it has never seen, and sp/win_rate is inflated.
            "sp/opp_piloted_frac": stats.get("sp_opp_piloted_frac"),
            # Should track --opp-expert-share.  0.0 means this branch never
            # met another branch's expert, whatever the share was set to.
            "sp/expert_opp_frac": stats.get("sp_expert_opp_frac"),
            "sp/own_seat_fraction": (
                stats["sp_samples"] / max(stats["sp_decisions"], 1)
                if stats.get("sp_samples") is not None else None),
            "mcts/step": step,
        }
        # Which opponent archetypes the adaptive sampler actually drew, as a
        # fraction of this iteration's games.
        n_games = max(stats["sp_games"], 1)
        for oid, count in stats.get("sp_opp_distribution", {}).items():
            payload[f"sp/opp_share/{_wb_key(oid)}"] = count / n_games
        # Self-play wall-clock breakdown (the PERF console line).
        for k, v in stats.get("perf", {}).items():
            payload[f"sp/perf/{k}"] = v
        self.log(payload, step)

    def log_train(self, stats: dict, step: int) -> None:
        self.log({"train/" + k: v for k, v in stats.items()
                  if isinstance(v, (int, float))}, step)

    def log_eval(self, league_result: dict, n_champions: int, step: int) -> None:
        results = league_result["results"]
        wr_list = [r["win_rate"] for r in results.values()]
        payload = {
            "eval/avg_wr": sum(wr_list) / max(len(wr_list), 1),
            "eval/min_wr": min(wr_list),
            "eval/max_wr": max(wr_list),
            "eval/n_opponents": len(results),
            "eval/n_champions": n_champions,
            "eval/passed": float(league_result["passed"]),
            "eval/n_passed": float(sum(r["passed"] for r in results.values())),
            "mcts/step": step,
        }
        # Per-opponent curves.  Champion names carry their game count, so each
        # champion gets its own series; IL_baseline is the one stable series
        # across the whole run and is worth reading on its own.
        for name, r in results.items():
            key = _wb_key(name)
            payload[f"eval/wr/{key}"] = r["win_rate"]
            payload[f"eval/wins/{key}"] = r["wins"]
            payload[f"eval/losses/{key}"] = r["losses"]
            payload[f"eval/draws/{key}"] = r["draws"]
        if "IL_baseline" in results:
            payload["eval/wr_vs_il"] = results["IL_baseline"]["win_rate"]
        self.log(payload, step)

    def log_elo(self, elo: "EloTracker", new_name: str, threshold: float,
                step: int) -> None:
        """The ELO gate: the candidate's rating against the pool it must beat."""
        best_name, best_elo = elo.best()
        payload = {
            "elo/rating": elo.ratings.get(new_name, elo.initial),
            "elo/threshold": threshold,
            "elo/best": best_elo,
            "elo/margin": elo.ratings.get(new_name, elo.initial) - threshold,
            "elo/n_models": len(elo.ratings),
            "elo/passed": float(
                elo.ratings.get(new_name, elo.initial) >= threshold),
            "mcts/step": step,
        }
        payload["elo/is_best"] = float(best_name == new_name)
        for name, rating in elo.ratings.items():
            payload[f"elo/rating/{_wb_key(name)}"] = rating
        self.log(payload, step)

    def log_model(self, policy, step: int) -> None:
        if not self.active:
            return
        payload: dict = {"mcts/step": step}
        for name, p in policy.named_parameters():
            n = p.data.numel()
            payload[f"model/{name}_mean"] = p.data.mean().item()
            payload[f"model/{name}_std"] = \
                p.data.std(unbiased=False).item() if n > 1 else 0.0
            if p.grad is not None:
                payload[f"grad/{name}_mean"] = p.grad.mean().item()
                payload[f"grad/{name}_std"] = \
                    p.grad.std(unbiased=False).item() if n > 1 else 0.0
        self._run.log(payload)

    def finish(self) -> None:
        if self.active:
            self._run.finish()


def _sample_weighted(
    opp_decks: list[dict],
    rng: np.random.Generator,
    weights: dict[int, float] | None = None,
) -> dict:
    """Sample an opponent, optionally with custom relative weights."""
    if weights is None:
        return opp_decks[int(rng.integers(0, len(opp_decks)))]
    w = np.array([weights.get(d["id"], 1.0) for d in opp_decks], dtype=np.float64)
    w = w / w.sum()
    return opp_decks[int(rng.choice(len(opp_decks), p=w))]


def _load_opp_decks(data_dir: Path, all_archetypes: bool = True) -> list[dict]:
    from ptcg_rl.search import load_opp_archetype_decks
    return load_opp_archetype_decks(str(data_dir), all_archetypes=all_archetypes)


def _save_elo(out_dir: Path, elo: EloTracker) -> None:
    """Persist ELO ratings to ``out_dir / elo_ratings.json``."""
    payload = {
        "ratings": elo.ratings,
        "fixed": sorted(elo.fixed),
    }
    path = out_dir / "elo_ratings.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("ELO ratings saved to %s", path)


def _load_elo(out_dir: Path, elo: EloTracker) -> bool:
    """Restore ELO ratings from ``out_dir / elo_ratings.json``.

    Returns True if a saved file was found and loaded.
    """
    path = out_dir / "elo_ratings.json"
    if not path.exists():
        return False
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load ELO ratings: %s", e)
        return False
    saved_ratings: dict = payload.get("ratings", {})
    saved_fixed: list = payload.get("fixed", [])
    for name, rating in saved_ratings.items():
        elo.ratings[name] = float(rating)
    for name in saved_fixed:
        elo.fixed.add(name)
    logger.info(
        "ELO ratings loaded from %s: %d models, %d fixed",
        path, len(saved_ratings), len(saved_fixed),
    )
    return True


def elo_key(ckpt: Path) -> str:
    """The :class:`EloTracker` key for a checkpoint *file*.

    ``EloTracker`` is keyed by bare model names — ``elo_name`` registers
    ``ckpt-mcts-champion-001400`` and ``_discover_il_checkpoints`` registers
    ``ckpt-best`` — so the key is the stem, **without** ``.pt``.

    Three call sites used ``ckpt.stem + ckpt.suffix`` instead, which put the
    suffix back on and therefore matched nothing.  Every lookup fell to its
    default of 0, so ``_find_best_champion`` saw a flat field and returned
    whichever file ``glob`` happened to yield first: a run with a 1553-rated
    champion on disk auto-resumed from a 1522-rated one and logged "ELO: 0".
    Nothing raises — a missing key is a default, not an error — and the
    leaderboard printed alongside it was correct, which is what made the
    inconsistency look like corrupt ELO state rather than a lookup bug.
    """
    return ckpt.stem


def _find_best_champion(
    champion_ckpts: list[Path], elo: EloTracker,
) -> Path | None:
    """Return the champion checkpoint with the highest ELO rating.

    Returns None if *champion_ckpts* is empty or none have a recorded rating.
    """
    best_ckpt = None
    best_elo = -float("inf")
    for ckpt in champion_ckpts:
        if not ckpt.exists():
            continue
        rating = elo.ratings.get(elo_key(ckpt), 0.0)
        logger.debug("  champion %s: ELO %.0f", ckpt.name, rating)
        if rating > best_elo:
            best_elo = rating
            best_ckpt = ckpt
    return best_ckpt


def _cosine_lr(step: int, total: int, peak: float, floor: float, warmup: int) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup:
        return peak * step / max(warmup, 1)
    if step >= total:
        return floor
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (peak - floor) * (1.0 + np.cos(np.pi * progress))


def _save_ckpt(
    out_dir: Path, policy: Any, tag: str, games: int, win_rate: float | None,
    deck_record: dict,
) -> None:
    """Write a champion.  *deck_record* is required — see `require_deck_record`.

    `main` builds and validates it before self-play starts, so reaching the
    raise here means a caller bypassed that, not that a run wasted its compute.
    """
    import torch as _torch

    from ptcg_il.deck import DECK_KEY, require_deck_record

    require_deck_record(deck_record, f"_save_ckpt(tag={tag!r})")
    ckpt = {
        "model_state_dict": {k: v.cpu() for k, v in policy.state_dict().items()},
        "config": dict(getattr(policy, "config", {})),
        "mcts_games": games,
        "mcts_win_rate": win_rate,
        "tag": tag,
        DECK_KEY: deck_record,
    }
    path = out_dir / f"ckpt-mcts-{tag}.pt"
    _torch.save(ckpt, path)
    logger.info("Saved %s", path)


# ── ELO Rating System ────────────────────────────────────────────────────


class EloTracker:
    """ELO ratings across all models (IL checkpoints + saved champions).

    ``ckpt-best`` is the fixed reference at 1500 — its rating never changes.
    """

    def __init__(self, k: float = 32.0, initial: float = 1500.0):
        self.k = k; self.initial = initial
        self.ratings: dict[str, float] = {}
        self.fixed: set[str] = set()  # models whose ELO is frozen (reference)

    def add_model(self, name: str, fixed: bool = False) -> None:
        self.ratings[name] = self.initial
        if fixed:
            self.fixed.add(name)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, a: str, b: str, wins: int, losses: int) -> tuple[float, float]:
        ra, rb = self.ratings.get(a, self.initial), self.ratings.get(b, self.initial)
        total = wins + losses
        if total == 0: return ra, rb
        delta = self.k * (wins / total - self.expected(ra, rb))
        if a not in self.fixed:
            self.ratings[a] = ra + delta
        if b not in self.fixed:
            self.ratings[b] = rb - delta
        return self.ratings.get(a, ra), self.ratings.get(b, rb)

    def top_pct(self, pct: float = 0.70) -> float:
        """Rating above which a model is in the top (100-pct)%."""
        vals = sorted(self.ratings.values())
        idx = max(0, int(len(vals) * pct))
        return vals[idx]

    def best(self) -> tuple[str, float]:
        if not self.ratings: return ("?", self.initial)
        name = max(self.ratings, key=self.ratings.get)
        return name, self.ratings[name]

    def leaderboard(self) -> str:
        ranked = sorted(self.ratings.items(), key=lambda x: -x[1])
        return "\n".join(f"  {n:<35} {r:>6.0f}  {'[FIXED]' if n in self.fixed else ''}"
                         for n, r in ranked)


if __name__ == "__main__":
    sys.exit(main())
