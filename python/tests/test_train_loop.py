"""Tests for training loop components: optimizer, scheduler, EMA, train_step,
checkpointing, eval, and dataset integration.

Uses a tiny Policy (D=32, layers=1) and synthetic shards to keep tests fast.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ptcg_il.model.policy import Policy
from ptcg_il.train.checkpoint import (
    build_submission_bundle,
    load_checkpoint,
    save_checkpoint,
)
from ptcg_il.train.dataset import ShardDataset, collate_fn
from ptcg_il.train.eval import offline_eval
from ptcg_il.train.logger import WandbLogger
from ptcg_il.train.loop import (
    _EMA,
    _no_decay,
    create_optimizer,
    create_schedule,
    train_step,
)

# ============================================================
# Tiny model factory
# ============================================================


def _tiny_policy() -> Policy:
    """Create a tiny Policy (D=32, layers=1) for fast tests."""
    return Policy(V=16, A=8, D=32, heads=4, layers=1, ff=64)


# ============================================================
# Synthetic data for training tests
# ============================================================


def _synthetic_shard_sample() -> dict[str, np.ndarray]:
    """One synthetic sample with shapes matching the featurizer contract."""
    # Capacities from A.1
    P_MAX, H_MAX, SUM, D_MAX, PZ_MAX = 12, 30, 2, 60, 6
    L_STATE, O_MAX = 46, 64
    F_GLOBAL, F_POKE, F_HAND, F_SUM, F_OPT = 93, 26, 2, 11, 6

    return {
        # State — card ids
        "poke_card_id": np.zeros(P_MAX, dtype=np.int64),
        "hand_card_id": np.zeros(H_MAX, dtype=np.int64),
        "stadium_card_id": np.zeros(1, dtype=np.int64),
        "context_card_id": np.zeros(1, dtype=np.int64),
        "effect_card_id": np.zeros(1, dtype=np.int64),
        "discard_ids": np.zeros((SUM, D_MAX), dtype=np.int64),
        "prize_ids": np.zeros((SUM, PZ_MAX), dtype=np.int64),
        # State — dense features
        "cls_feat": np.random.randn(F_GLOBAL).astype(np.float32) * 0.1,
        "poke_feat": np.random.randn(P_MAX, F_POKE).astype(np.float32) * 0.1,
        "hand_feat": np.random.randn(H_MAX, F_HAND).astype(np.float32) * 0.1,
        "sum_feat": np.random.randn(SUM, F_SUM).astype(np.float32) * 0.1,
        "stadium_present": np.array([1.0], dtype=np.float32),
        # State — categorical token attributes
        "tok_type": np.zeros(L_STATE, dtype=np.int64),
        "tok_owner": np.zeros(L_STATE, dtype=np.int64),
        "tok_zone": np.zeros(L_STATE, dtype=np.int64),
        "tok_mask": np.zeros(L_STATE, dtype=bool),
        # State — make 10 pokemon slots active for plausible attention
        # Action — option references
        "opt_type": np.zeros(O_MAX, dtype=np.int64),
        "opt_src_idx": np.full(O_MAX, -1, dtype=np.int64),
        "opt_tgt_idx": np.full(O_MAX, -1, dtype=np.int64),
        "opt_card_id": np.zeros(O_MAX, dtype=np.int64),
        "opt_attack_idx": np.zeros(O_MAX, dtype=np.int64),
        "opt_scalar": np.zeros((O_MAX, F_OPT), dtype=np.float32),
        "opt_mask": np.zeros(O_MAX, dtype=bool),
        # Labels
        "action_idx": np.full(O_MAX, -1, dtype=np.int64),
        "action_len": np.array(1, dtype=np.int64),
        "minCount": np.array(1, dtype=np.int64),
        "maxCount": np.array(1, dtype=np.int64),
        "sel_type": np.array(0, dtype=np.int64),
        "sel_ctx": np.array(0, dtype=np.int64),
        "value_target": np.array(1.0, dtype=np.float32),
        # Masks
        "discard_mask": np.zeros((SUM, D_MAX), dtype=bool),
    }


def _configure_sample(
    sample: dict[str, np.ndarray],
    *,
    sel_ctx: int = 0,
    won: bool = True,
    max_count: int = 1,
    n_options: int = 4,
) -> dict[str, np.ndarray]:
    """Set realistic values on a synthetic sample."""
    s = {k: v.copy() for k, v in sample.items()}

    # Activate some state tokens
    s["tok_mask"][:10] = True  # 10 pokemon slots
    s["tok_mask"][13:18] = True  # 5 hand cards
    s["tok_mask"][43:45] = True  # 2 summary
    s["tok_mask"][45] = True  # stadium

    # Token types
    s["tok_type"][:10] = 1  # POKE
    s["tok_type"][13:18] = 2  # HAND
    s["tok_type"][43:45] = 3  # SUMMARY
    s["tok_type"][45] = 4  # STADIUM

    # Options
    s["opt_mask"][:n_options] = True
    for i in range(n_options):
        s["opt_type"][i] = 7  # PLAY
        s["opt_src_idx"][i] = 13 + i  # hand slot

    # Action
    s["action_idx"][0] = np.random.randint(0, n_options)  # expert pick
    s["action_len"] = np.array(1, dtype=np.int64)
    s["minCount"] = np.array(1, dtype=np.int64)
    s["maxCount"] = np.array(max_count, dtype=np.int64)
    s["sel_ctx"] = np.array(sel_ctx, dtype=np.int64)
    s["sel_type"] = np.array(min(sel_ctx, 10), dtype=np.int64)
    s["value_target"] = np.array(1.0 if won else -1.0, dtype=np.float32)

    # Embeddings need real card ids for some tokens
    for i in range(10):
        s["poke_card_id"][i] = 2  # some real card
    for i in range(5):
        s["hand_card_id"][i] = 2

    return s


def _build_tiny_data(num_samples: int = 32) -> Path:
    """Create a temporary data dir with one shard + meta.parquet."""
    tmp = tempfile.mkdtemp()
    data_dir = Path(tmp)
    shards_dir = data_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    base = _synthetic_shard_sample()
    samples = [_configure_sample(base, sel_ctx=i % 5, won=(i % 2 == 0)) for i in range(num_samples)]

    stacked = {}
    keys = sorted(samples[0].keys())
    for k in keys:
        stacked[k] = np.stack([s[k] for s in samples], axis=0)
    np.savez_compressed(shards_dir / "train-00000.npz", **stacked)
    np.savez_compressed(shards_dir / "val-00000.npz", **stacked)

    # Meta — train rows point to train shard, val to val shard
    meta_rows = []
    half = num_samples // 2
    for i in range(num_samples):
        sp = "train" if i < half else "val"
        meta_rows.append({
            "sample_uid": f"ep_{i}_0_0",
            "shard": f"{sp}-00000.npz",
            "row": i if i < half else i - half,
            "episode_id": f"ep_{i}",
            "player": 0,
            "team": "expert_A",
            "archetype_self": 0,
            "archetype_opp": 0,
            "sel_type": 0,
            "sel_ctx": i % 5,
            "minCount": 1,
            "maxCount": 1,
            "won": (i % 2 == 0),
        })
    pd.DataFrame(meta_rows).to_parquet(data_dir / "meta.parquet", index=False)

    # Also write minimal vocab/archetypes for checkpoint bundle tests
    data_dir.joinpath("vocab.json").write_text(json.dumps({
        "id_remap": {}, "attack_remap": {},
        "N_VOCAB": 10, "V": 12, "A": 8,
    }))
    data_dir.joinpath("archetypes.json").write_text(json.dumps({
        "self_ids": [0], "opp_ids": [0],
        "archetypes": [{"id": 0, "representative": [2]*60, "frequency": 100}],
    }))
    data_dir.joinpath("deck.csv").write_text("2\n" * 60)

    return data_dir


# ============================================================
# Tests — _no_decay
# ============================================================


class TestNoDecay:
    """Verify per-param no-decay set logic."""

    def test_bias_no_decay(self):
        """Bias parameters should not be decayed."""
        p = torch.nn.Parameter(torch.zeros(4))
        assert _no_decay("encoder.enc.layers.0.self_attn.in_proj_bias", p)

    def test_layernorm_no_decay(self):
        """LayerNorm parameters should not be decayed."""
        p = torch.nn.Parameter(torch.ones(32))
        assert _no_decay("encoder.enc.layers.0.norm1.weight", p)

    def test_embedding_no_decay(self):
        """Embedding weights should not be decayed."""
        p = torch.nn.Parameter(torch.zeros(16, 32))
        assert _no_decay("embed.card.id_emb.weight", p)

    def test_null_token_no_decay(self):
        """null_token should not be decayed."""
        p = torch.nn.Parameter(torch.zeros(32))
        assert _no_decay("pointer.null_token", p)

    def test_no_stadium_no_decay(self):
        """no_stadium should not be decayed."""
        p = torch.nn.Parameter(torch.zeros(32))
        assert _no_decay("embed.no_stadium", p)

    def test_linear_weight_decayed(self):
        """Regular Linear weight should be decayed."""
        p = torch.nn.Parameter(torch.zeros(32, 64))
        assert not _no_decay("cls_mlp.0.weight", p)


# ============================================================
# Tests — create_optimizer
# ============================================================


class TestCreateOptimizer:
    def test_creates_two_param_groups(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy, peak_lr=3e-4)
        assert len(opt.param_groups) == 2

    def test_no_decay_group_has_wd_zero(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy, peak_lr=3e-4, weight_decay=0.01)
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_decay_group_has_nonzero_wd(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy, peak_lr=3e-4, weight_decay=0.01)
        assert opt.param_groups[0]["weight_decay"] == 0.01


# ============================================================
# Tests — create_schedule
# ============================================================


class TestCreateSchedule:
    def test_warmup_lr_increases(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy)
        sched = create_schedule(opt, total_steps=500, warmup=100, peak_lr=3e-4)

        lr0 = sched.get_last_lr()[0]
        opt.step(); sched.step()
        lr1 = sched.get_last_lr()[0]
        # After 1 step of warmup, LR should increase
        assert lr1 > lr0

    def test_after_warmup_at_peak(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy)
        sched = create_schedule(opt, total_steps=500, warmup=100, peak_lr=3e-4)

        for _ in range(100):
            opt.step(); sched.step()
        lr = sched.get_last_lr()[0]
        assert np.isclose(lr, 3e-4, rtol=0.1)

    def test_cosine_decay(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy)
        sched = create_schedule(opt, total_steps=500, warmup=100, peak_lr=3e-4, min_lr=3e-5)

        for _ in range(100):
            opt.step(); sched.step()
        lr_warmup_end = sched.get_last_lr()[0]

        for _ in range(300):
            opt.step(); sched.step()
        lr_late = sched.get_last_lr()[0]

        assert lr_late < lr_warmup_end  # cosine decay should lower LR


# ============================================================
# Tests — _EMA
# ============================================================


class TestEMA:
    def test_state_dict_roundtrip(self):
        policy = _tiny_policy()
        ema = _EMA(policy, decay=0.999)
        policy.embed.card.id_emb.weight.data.fill_(1.0)
        ema.update(policy)
        sd = ema.state_dict()
        ema2 = _EMA(policy, decay=0.999)
        ema2.load_state_dict(sd)
        # Shadow should be updated
        for n, s in ema2.shadow.items():
            assert torch.equal(s, ema.shadow[n])

    def test_apply_copies_weights(self):
        policy = _tiny_policy()
        # Set initial weights to known value before creating EMA
        for p in policy.parameters():
            p.data.fill_(1.0)
        ema = _EMA(policy, decay=0.999)
        # Modify original and update
        policy.embed.card.id_emb.weight.data.fill_(5.0)
        ema.update(policy)
        # Change policy back
        policy.embed.card.id_emb.weight.data.fill_(0.0)
        # Apply EMA back — should restore weighted version
        ema.apply(policy)
        val = policy.embed.card.id_emb.weight.data.mean().item()
        # After one update with decay 0.999: shadow = 0.999*1 + 0.001*5 ≈ 1.004
        # After apply: should be ~1.004, which is > 0
        assert val > 0.0

    def test_multiple_updates_converge(self):
        policy = _tiny_policy()
        ema = _EMA(policy, decay=0.0)  # No smoothing → shadow = new value each step
        policy.embed.card.id_emb.weight.data.fill_(3.0)
        ema.update(policy)
        policy.embed.card.id_emb.weight.data.fill_(0.0)
        ema.apply(policy)
        val = policy.embed.card.id_emb.weight.data.mean().item()
        assert abs(val - 3.0) < 1e-4  # with decay=0, shadow = new value


# ============================================================
# Tests — train_step
# ============================================================


class TestTrainStep:
    def test_single_select_forward(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy)
        ema = _EMA(policy)
        data_dir = _build_tiny_data(32)
        ds = ShardDataset(data_dir, split="train")
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))

        device = torch.device("cpu")
        batch_gpu = {k: v.to(device) for k, v in batch.items() if k != "encoder_padding_mask"}

        metrics = train_step(policy, batch_gpu, opt, ema, device, grad_scaler=None)
        assert "loss" in metrics
        assert "ce" in metrics
        assert "value_mse" in metrics
        assert "grad_norm" in metrics
        assert metrics["loss"] > 0

    def test_train_step_reduces_loss(self):
        torch.manual_seed(42)
        np.random.seed(42)
        policy = _tiny_policy()
        opt = create_optimizer(policy)
        ema = _EMA(policy)
        data_dir = _build_tiny_data(64)
        ds = ShardDataset(data_dir, split="train")
        loader = DataLoader(ds, batch_size=16, collate_fn=collate_fn)
        batch = next(iter(loader))

        device = torch.device("cpu")
        batch_gpu = {k: v.to(device) for k, v in batch.items() if k != "encoder_padding_mask"}

        # Run multiple steps on the same batch — loss should decrease
        losses = []
        for _ in range(20):
            metrics = train_step(policy, batch_gpu, opt, ema, device, grad_scaler=None)
            losses.append(metrics["loss"])

        # Loss should have decreased after 20 steps
        assert losses[0] > 0
        avg_early = sum(losses[:5]) / 5
        avg_late = sum(losses[-5:]) / 5
        assert avg_late < avg_early, f"Loss did not decrease: early={avg_early:.4f}, late={avg_late:.4f}"


# ============================================================
# Tests — checkpoint
# ============================================================


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy)
        ema = _EMA(policy)
        sched = create_schedule(opt, total_steps=500)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(policy, opt, sched, ema, step=100, save_dir=tmp)
            assert path.exists()

            ckpt = load_checkpoint(path, device="cpu")
            assert ckpt["step"] == 100
            assert "model_state_dict" in ckpt
            assert "optimizer_state_dict" in ckpt
            assert "scheduler_state_dict" in ckpt
            assert "ema_state_dict" in ckpt
            assert "rng_state" in ckpt

    def test_tag_in_filename(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy)
        ema = _EMA(policy)
        sched = create_schedule(opt, total_steps=500)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(policy, opt, sched, ema, step=2000, save_dir=tmp, tag="best")
            assert "best" in str(path)

    def test_build_submission_bundle(self):
        policy = _tiny_policy()
        opt = create_optimizer(policy)
        ema = _EMA(policy)
        sched = create_schedule(opt, total_steps=500)
        data_dir = _build_tiny_data(32)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = save_checkpoint(policy, opt, sched, ema, step=100, save_dir=tmp, tag="best")
            ema.apply(policy)  # Apply EMA before saving
            ckpt_path = save_checkpoint(policy, opt, sched, ema, step=100, save_dir=tmp, tag="best")

            sub_dir = build_submission_bundle(
                ckpt_path, data_dir, tmp,
            )
            assert sub_dir.exists()
            assert (sub_dir / "weights.pt").exists()
            assert (sub_dir / "vocab.json").exists()
            assert (sub_dir / "archetypes.json").exists()
            assert (sub_dir / "deck.csv").exists()


# ============================================================
# Tests — offline_eval
# ============================================================


class TestOfflineEval:
    def test_returns_metrics_dict(self):
        policy = _tiny_policy()
        data_dir = _build_tiny_data(32)
        ds = ShardDataset(data_dir, split="val")
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

        device = torch.device("cpu")
        metrics = offline_eval(policy, loader, device, max_batches=2)

        assert isinstance(metrics, dict)
        assert "val/top1_micro" in metrics
        assert "val/top1_macro" in metrics
        assert "val/top3_micro" in metrics
        assert "val/value_mse" in metrics
        assert "val/value_auc" in metrics

    def test_top1_between_zero_and_one(self):
        policy = _tiny_policy()
        data_dir = _build_tiny_data(32)
        ds = ShardDataset(data_dir, split="val")
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

        device = torch.device("cpu")
        metrics = offline_eval(policy, loader, device, max_batches=2)
        assert 0.0 <= metrics["val/top1_micro"] <= 1.0

    def test_per_sel_ctx_table(self):
        policy = _tiny_policy()
        data_dir = _build_tiny_data(32)
        ds = ShardDataset(data_dir, split="val")
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

        device = torch.device("cpu")
        metrics = offline_eval(policy, loader, device, max_batches=2)
        assert "val/top1_by_sel_ctx" in metrics
        assert isinstance(metrics["val/top1_by_sel_ctx"], list)

    def test_ema_restored_after_eval(self):
        policy = _tiny_policy()
        ema = _EMA(policy)
        data_dir = _build_tiny_data(32)
        ds = ShardDataset(data_dir, split="val")
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

        # Record original weights
        orig_w = policy.embed.card.id_emb.weight.data.clone()

        device = torch.device("cpu")
        _ = offline_eval(policy, loader, device, ema=ema, max_batches=2)

        # Weights should be restored
        restored_w = policy.embed.card.id_emb.weight.data
        assert torch.equal(orig_w, restored_w)


# ============================================================
# Tests — WandbLogger (no-op mode)
# ============================================================


class TestWandbLogger:
    def test_creates_in_noop_mode(self):
        """wandb may not be installed or configured. Logger should not crash."""
        logger = WandbLogger(mode="disabled")
        # In disabled mode, logger should not crash regardless of active state
        assert isinstance(logger.active, bool)

    def test_log_train_noop(self):
        logger = WandbLogger(mode="disabled")
        # Should not raise
        logger.log_train(step=1, loss=0.5, ce=0.4, value_mse=0.1,
                         grad_norm=0.2, lr=3e-4, samples_per_sec=100.0)

    def test_log_eval_noop(self):
        logger = WandbLogger(mode="disabled")
        logger.log_eval(step=100, **{
            "val/top1_micro": 0.8,
            "val/top1_macro": 0.7,
            "val/top3_micro": 0.9,
            "val/value_mse": 0.3,
            "val/value_auc": 0.65,
            "val/top1_by_sel_ctx": [(0, 0.9, 10), (1, 0.7, 5)],
            "val/top1_by_sel_type": [(0, 0.85, 15)],
            "val/best_top1_macro": 0.0,
        })

    def test_mark_best_noop(self):
        logger = WandbLogger(mode="disabled")
        logger.mark_best(step=1000, macro_top1=0.75)

    def test_finish_noop(self):
        logger = WandbLogger(mode="disabled")
        logger.finish()
