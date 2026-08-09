"""ELO calibration — round-robin tournament with ckpt-best anchored at 1500.

Two modes:

``round-robin`` (default)
    Every model plays every other model.  ``--games`` controls games per
    pairing (default 10, split evenly across two seats).  More stable ratings
    because each model is evaluated against N−1 opponents instead of one.

``reference``
    Each model plays only ckpt-best (the old behavior).  ``--games`` defaults
    to 100 per model since there is only one opponent.

Usage::

    cd python
    # Round-robin (default): 10 games per pair, every model plays everyone
    uv run python -m ptcg_rl.elo_calibrate \
        --il-ckpt checkpoints_a0/ckpt-best.pt \
        --data-dir data --out-dir checkpoints_a0_mcts

    # Reference mode (old behavior): 100 games vs ckpt-best only
    uv run python -m ptcg_rl.elo_calibrate \
        --il-ckpt checkpoints_a0/ckpt-best.pt \
        --data-dir data --out-dir checkpoints_a0_mcts \
        --mode reference --games 100

Discovers all .pt files in the IL checkpoint directory and any MCTS
champions in *out-dir* and writes ELO ratings to
``out-dir/elo_ratings.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
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


# ── Reuse from mcts_train ────────────────────────────────────────────────
def _load_vocab(data_dir: Path) -> dict:
    from ptcg_il.featurizer import normalize_vocab
    with open(data_dir / "vocab.json") as f:
        return normalize_vocab(json.load(f))


def _load_archetypes(data_dir: Path) -> dict:
    with open(data_dir / "archetypes.json") as f:
        return json.load(f)



def _static_tables(data_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, fname in (("card_static_table", "card_static_table.npy"),
                       ("attack_static_table", "attack_static_table.npy")):
        path = data_dir / fname
        out[key] = torch.from_numpy(np.load(path)).float() if path.exists() else None
    return out


def _load_lenient(policy: Any, model_sd: dict, ckpt_sd: dict) -> int:
    """Load *ckpt_sd* into *policy*, handling size-mismatched ``*.static`` params.

    For any parameter whose checkpoint shape differs from the model, only
    the overlapping prefix is copied.  Everything else is loaded as-is.
    Missing belief-head keys are tolerated.
    """
    loaded = 0
    for key, model_param in model_sd.items():
        ckpt_param = ckpt_sd.get(key)
        if ckpt_param is None:
            if key.startswith("belief_heads."):
                continue
            continue
        if ckpt_param.shape != model_param.shape:
            slices = tuple(
                slice(0, min(cs, ms))
                for cs, ms in zip(ckpt_param.shape, model_param.shape)
            )
            model_param[slices].copy_(ckpt_param[slices])
            logger.debug("  _load_lenient: %s %s → %s (partial)",
                         key, list(ckpt_param.shape), list(model_param.shape))
        else:
            model_param.copy_(ckpt_param)
        loaded += 1
    return loaded


def _load_policy(ckpt_path: str, data_dir: Path, device: torch.device) -> Any | None:
    """Load a policy from a checkpoint (pure-feature format).

    Returns None if the checkpoint is missing a ``config`` record.
    Loads ``all_card_feat`` from the data directory for the belief heads.
    """
    from ptcg_il.model.cards import F_CARD
    from ptcg_il.model.policy import load_policy_state, policy_from_config
    from ptcg_il.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(ckpt_path, device="cpu")
    config = ckpt.get("config") or {}
    if not config:
        logger.warning("  SKIP %s — no config record", Path(ckpt_path).name)
        return None

    # Load all-card feature matrix for belief heads
    all_card_feat = None
    ecf_path = data_dir / "engine_card_features.npy"
    if ecf_path.exists():
        ecf = np.load(ecf_path, allow_pickle=True).item()
        # Build [n_cards, F_CARD] matrix sorted by card id
        max_id = max(ecf.keys()) if ecf else 0
        all_card_feat = torch.zeros(max_id + 1, F_CARD)
        for cid, feat in ecf.items():
            all_card_feat[int(cid)] = torch.from_numpy(np.asarray(feat, dtype=np.float32))

    policy = policy_from_config(config, all_card_feat=all_card_feat)
    model_sd = policy.state_dict()
    ckpt_sd = ckpt["model_state_dict"]
    try:
        load_policy_state(policy, ckpt_sd)
    except RuntimeError:
        loaded = _load_lenient(policy, model_sd, ckpt_sd)
        logger.info("  Lenient load: %d/%d params loaded", loaded, len(model_sd))

    policy.to(device)
    return policy


def _fixed_deck(data_dir: Path) -> list[int]:
    with open(data_dir / "archetypes.json") as f:
        return [int(c) for c in json.load(f)["fixed_deck"]]


def _load_opp_decks(data_dir: Path) -> list[dict]:
    from ptcg_rl.search import load_opp_archetype_decks
    return load_opp_archetype_decks(str(data_dir), all_archetypes=True)


# ── Checkpoint discovery ─────────────────────────────────────────────────

def _discover_all_checkpoints(il_ckpt_path: str, out_dir: Path) -> list[tuple[str, str]]:
    """Discover all .pt checkpoints for ELO calibration.

    Returns ``[(name, path), ...]``.  ``ckpt-best`` is always first.
    """
    # ckpt_dir = Path(il_ckpt_path).resolve().parent
    # result: list[tuple[str, str]] = []
    result = [(str(i.name), str(i)) for i in Path(il_ckpt_path).glob('*.pt')]
    result.extend([(str(i.name), str(i)) for i in Path(out_dir).glob('*.pt')])
    # # ckpt-best first (the reference)
    # best = ckpt_dir / "ckpt-best.pt"
    # if best.exists():
    #     result.append(("ckpt-best", str(best)))

    # # Step checkpoints from IL directory
    # for p in sorted(ckpt_dir.glob("ckpt-step-*.pt")):
    #     result.append((p.stem, str(p)))

    # # MCTS champions from output directory
    # for p in sorted(out_dir.glob("ckpt-mcts-champion-*.pt")):
    #     result.append((p.stem, str(p)))

    # # ckpt-last from IL directory
    # last = ckpt_dir / "ckpt-last.pt"
    # if last.exists():
    #     result.append(("ckpt-last", str(last)))
    print(f'result : {result}')
    return result


# ── ELO ──────────────────────────────────────────────────────────────────

class EloTracker:
    """Minimal ELO tracker with fixed-reference support."""

    def __init__(self, k: float = 32.0, initial: float = 1500.0):
        self.k = k; self.initial = initial
        self.ratings: dict[str, float] = {}
        self.fixed: set[str] = set()

    def add_model(self, name: str, fixed: bool = False) -> None:
        self.ratings[name] = self.initial
        if fixed:
            self.fixed.add(name)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, a: str, b: str, wins: int, losses: int) -> tuple[float, float]:
        ra = self.ratings.get(a, self.initial)
        rb = self.ratings.get(b, self.initial)
        total = wins + losses
        if total == 0:
            return ra, rb
        delta = self.k * (wins / total - self.expected(ra, rb))
        if a not in self.fixed:
            self.ratings[a] = ra + delta
        if b not in self.fixed:
            self.ratings[b] = rb - delta
        return self.ratings.get(a, ra), self.ratings.get(b, rb)


# ── Evaluation ───────────────────────────────────────────────────────────

def _eval_pair(
    model_a: Any,
    model_b: Any,
    vocab: dict,
    fixed_deck: list[int],
    opp_decks: list[dict],
    device: torch.device,
    n_games: int,
    seed: int,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
    evolution_map: dict | None = None,
) -> tuple[int, int]:
    """Play *n_games* between model_a (our player) and model_b (opponent).

    Returns ``(wins_for_a, losses_for_a)``.
    """
    from ptcg_rl.rust_vec_env import RustVecEnv
    from ptcg_rl.rollout import PolicyActor

    model_a.eval()
    model_b.eval()

    actor_a = PolicyActor(model_a, vocab, device=str(device), greedy=True, seed=seed,
                          engine_card_features=engine_card_features,
                          engine_attack_features=engine_attack_features,
                          evolution_map=evolution_map)
    actor_b = PolicyActor(model_b, vocab, device=str(device), greedy=True, seed=seed + 1,
                          engine_card_features=engine_card_features,
                          engine_attack_features=engine_attack_features,
                          evolution_map=evolution_map)

    wins, losses = 0, 0
    n_done = 0
    rng = np.random.default_rng(seed)

    with RustVecEnv(
        deck_self=fixed_deck,
        deck_opp=opp_decks[0]["deck"],
        n_envs=4, our_player=0, seed=seed,
    ) as env:
        while n_done < n_games:
            pending = env.poll()
            if not pending:
                for t in env.drain():
                    if t.reward > 0:
                        wins += 1
                    elif t.reward < 0:
                        losses += 1
                    n_done += 1
                if not pending:
                    time.sleep(0.001)
                continue

            replies = [None] * len(pending)
            ours_reqs, theirs_reqs = [], []
            ours_idx, theirs_idx = [], []

            for pi, p in enumerate(pending):
                obs = json.loads(p["obs_json"])
                if p["select_player"] == 0:
                    ours_reqs.append({"obs": obs, "actor": "ours"})
                    ours_idx.append(pi)
                else:
                    theirs_reqs.append({"obs": obs, "actor": "theirs"})
                    theirs_idx.append(pi)

            if ours_reqs:
                for ri, rep in enumerate(actor_a(ours_reqs)):
                    replies[ours_idx[ri]] = rep
            if theirs_reqs:
                for ri, rep in enumerate(actor_b(theirs_reqs)):
                    replies[theirs_idx[ri]] = rep

            picks_list = [rep["picks"] for rep in replies if rep is not None]
            if picks_list:
                env.reply(picks_list)

            for t in env.drain():
                if t.reward > 0:
                    wins += 1
                elif t.reward < 0:
                    losses += 1
                n_done += 1

    return wins, losses


# ── Main ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptcg_rl.elo_calibrate",
        description="ELO calibration: round-robin tournament with ckpt-best anchored at 1500",
    )
    p.add_argument("--il-ckpt", type=str, required=True,
                   help="Path to ckpt-best.pt (the 1500 reference)")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="checkpoints_mcts",
                   help="Output directory for elo_ratings.json")
    p.add_argument("--mode", type=str, choices=("round-robin", "reference"),
                   default="round-robin",
                   help="Tournament mode: round-robin (default) or reference (vs ckpt-best only)")
    p.add_argument("--games", type=int, default=None,
                   help="Games per pairing (default: 10 round-robin, 100 reference)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p


# ── Tournament implementations ─────────────────────────────────────────────

def _eval_with_seats(
    model_a: Any, model_b: Any,
    name_a: str, name_b: str,
    vocab: dict, fixed_deck: list[int], opp_decks: list[dict],
    device: torch.device, games: int, seed: int,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
    evolution_map: dict | None = None,
) -> tuple[int, int]:
    """Play *games* between *model_a* and *model_b*, balanced across seats.

    Returns ``(wins_for_a, losses_for_a)`` where wins/losses are from
    model_a's perspective.  The seat-swap balances any deck-position
    advantage (self-deck vs archetype-deck).

    The wins/losses counting is correct regardless of which model is
    ``our_player`` in a given seat: ``_eval_pair`` returns
    ``(wins_for_our_player, losses_for_our_player)``, and we track
    model_a's wins explicitly.
    """
    per_seat = max(1, games // 2)
    wins_a = 0
    for seat in (0, 1):
        w, l = _eval_pair(
            model_a if seat == 0 else model_b,   # our_player
            model_b if seat == 0 else model_a,   # opponent
            vocab, fixed_deck, opp_decks, device,
            per_seat, seed + seat,
            engine_card_features=engine_card_features,
            engine_attack_features=engine_attack_features,
            evolution_map=evolution_map,
        )
        if seat == 0:
            wins_a += w      # a is our_player → w = a's wins
        else:
            wins_a += l      # b is our_player → l = b's losses = a's wins
    total = per_seat * 2
    return wins_a, total - wins_a


def _tournament_round_robin(
    all_ckpts: list[tuple[str, str]],
    vocab: dict, fixed_deck: list[int], opp_decks: list[dict],
    data_dir: Path, device: torch.device,
    games: int, seed: int, elo: EloTracker,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
    evolution_map: dict | None = None,
) -> None:
    """Round-robin: every model plays every other model.

    Models are loaded into a CPU-side cache so each checkpoint file is
    read and built only once.  They are moved to GPU one pair at a time.
    """
    # ── Load all models to CPU ──────────────────────────────────────────
    cpu_device = torch.device("cpu")
    models_cpu: list[tuple[str, Any]] = []  # (name, model)
    for name, path in all_ckpts:
        model = _load_policy(path, data_dir, cpu_device)
        if model is None:
            logger.warning("  SKIP %s — failed to load", name)
            continue
        model.eval()
        models_cpu.append((name, model))
        if name not in elo.ratings:
            elo.add_model(name, fixed=(name == "ckpt-best"))

    n = len(models_cpu)
    if n < 2:
        logger.warning("Need at least 2 models for round-robin; got %d", n)
        return

    n_pairs = n * (n - 1) // 2
    logger.info("Round-robin: %d models → %d pairs, %d games/pair → ~%d total games",
                n, n_pairs, games, n_pairs * games)

    pair_idx = 0
    for i in range(n):
        name_i, model_i = models_cpu[i]
        model_i.to(device)
        logger.info("[%d/%d] %s", i + 1, n, name_i)

        for j in range(i + 1, n):
            name_j, model_j = models_cpu[j]
            model_j.to(device)

            pair_seed = seed + i * 1000 + j * 10
            wins_i, losses_i = _eval_with_seats(
                model_i, model_j, name_i, name_j,
                vocab, fixed_deck, opp_decks, device,
                games, pair_seed,
                engine_card_features=engine_card_features,
                engine_attack_features=engine_attack_features,
                evolution_map=evolution_map,
            )
            pair_idx += 1
            elo.update(name_i, name_j, wins_i, losses_i)
            wr = wins_i / max(wins_i + losses_i, 1)
            logger.info(
                "  [%d/%d] vs %s: %dW/%dL wr=%.3f → ELO %s=%.0f %s=%.0f",
                pair_idx, n_pairs, name_j, wins_i, losses_i, wr,
                name_i, elo.ratings[name_i],
                name_j, elo.ratings[name_j],
            )

            model_j.to("cpu")

        model_i.to("cpu")
        # Release reference so GC can free the model if memory is tight
        models_cpu[i] = (name_i, None)  # type: ignore[assignment]


def _tournament_reference(
    all_ckpts: list[tuple[str, str]],
    vocab: dict, fixed_deck: list[int], opp_decks: list[dict],
    data_dir: Path, device: torch.device,
    games: int, seed: int, elo: EloTracker,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
    evolution_map: dict | None = None,
) -> None:
    """Reference mode: each non-reference model plays only ckpt-best."""
    # Find ckpt-best path
    best_path = None
    for name, path in all_ckpts:
        if name == "ckpt-best":
            best_path = path
            break
    if best_path is None:
        logger.error("ckpt-best not found in discovered checkpoints")
        return

    ref_model = _load_policy(best_path, data_dir, device)
    if ref_model is None:
        logger.error("REFERENCE MODEL ckpt-best FAILED TO LOAD")
        return
    ref_model.eval()

    other_models = [(n, p) for n, p in all_ckpts if n != "ckpt-best"]
    total = len(other_models)
    logger.info("Reference mode: %d models vs ckpt-best, %d games each",
                total, games)

    for i, (name, path) in enumerate(other_models):
        model = _load_policy(path, data_dir, device)
        if model is None:
            logger.warning("  SKIP %s — failed to load", name)
            continue
        model.eval()
        if name not in elo.ratings:
            elo.add_model(name)

        logger.info("[%d/%d] %s vs ckpt-best (%d games)", i + 1, total, name, games)

        wins_for_model, losses_for_model = _eval_with_seats(
            model, ref_model, name, "ckpt-best",
            vocab, fixed_deck, opp_decks, device,
            games, seed + i * 10,
            engine_card_features=engine_card_features,
            engine_attack_features=engine_attack_features,
            evolution_map=evolution_map,
        )
        wr = wins_for_model / max(wins_for_model + losses_for_model, 1)
        elo.update(name, "ckpt-best", wins_for_model, losses_for_model)
        logger.info("  %s: %dW/%dL wr=%.3f → ELO %.0f",
                    name, wins_for_model, losses_for_model, wr, elo.ratings[name])

        model.to("cpu")
        del model


# ── Main ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Resolve mode-dependent defaults
    if args.games is None:
        args.games = 10 if args.mode == "round-robin" else 100

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load shared artifacts ───────────────────────────────────────────
    logger.info("Loading artifacts from %s", data_dir)
    vocab = _load_vocab(data_dir)
    fixed_deck = _fixed_deck(data_dir)
    opp_decks = _load_opp_decks(data_dir)
    logger.info("Opponent decks: %d", len(opp_decks))

    # Load engine card/attack feature maps (required for pure-feature model)
    engine_card_features = None
    engine_attack_features = None
    ecf_path = data_dir / "engine_card_features.npy"
    eaf_path = data_dir / "engine_attack_features.npy"
    if ecf_path.exists():
        engine_card_features = np.load(ecf_path, allow_pickle=True).item()
        logger.info("Engine card features loaded (%d cards)", len(engine_card_features))
    if eaf_path.exists():
        engine_attack_features = np.load(eaf_path, allow_pickle=True).item()
        logger.info("Engine attack features loaded (%d attacks)", len(engine_attack_features))
    evo_path = data_dir / "evolution_map.npy"
    evolution_map = (
        np.load(evo_path, allow_pickle=True).item() if evo_path.exists() else None
    )
    if engine_card_features is None:
        logger.warning("No engine_card_features.npy — cards will get zero features!")

    # ── Discover checkpoints ────────────────────────────────────────────
    all_ckpts = _discover_all_checkpoints(args.il_ckpt, out_dir)
    logger.info("Discovered %d checkpoints (mode: %s, games/pair: %d)",
                len(all_ckpts), args.mode, args.games)
    for name, path in all_ckpts:
        logger.info("  %-45s  %s", name, path)

    # ── ELO tracker ─────────────────────────────────────────────────────
    elo = EloTracker()
    elo.add_model("ckpt-best", fixed=True)

    # ── Run tournament ──────────────────────────────────────────────────
    t_start = time.perf_counter()
    if args.mode == "round-robin":
        _tournament_round_robin(
            all_ckpts, vocab, fixed_deck, opp_decks,
            data_dir, device, args.games, args.seed, elo,
            engine_card_features=engine_card_features,
            engine_attack_features=engine_attack_features,
            evolution_map=evolution_map,
        )
    else:
        _tournament_reference(
            all_ckpts, vocab, fixed_deck, opp_decks,
            data_dir, device, args.games, args.seed, elo,
            engine_card_features=engine_card_features,
            engine_attack_features=engine_attack_features,
            evolution_map=evolution_map,
        )

    # ── Save ────────────────────────────────────────────────────────────
    payload = {"ratings": elo.ratings, "fixed": sorted(elo.fixed)}
    out_path = out_dir / "elo_ratings.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("ELO ratings saved to %s", out_path)

    # ── Leaderboard ─────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    ranked = sorted(elo.ratings.items(), key=lambda x: -x[1])
    print("\n" + "=" * 60)
    print("ELO Leaderboard")
    print("=" * 60)
    for rank, (name, rating) in enumerate(ranked, 1):
        marker = " [FIXED]" if name in elo.fixed else ""
        delta = rating - 1500
        print(f"  {rank:>3}. {name:<45} {rating:>6.0f}  ({delta:+.0f}){marker}")
    print(f"\n{len(ranked)} models  •  {fmt_dur(elapsed)}  •  saved to {out_path}")
    return 0


def fmt_dur(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


if __name__ == "__main__":
    sys.exit(main())
