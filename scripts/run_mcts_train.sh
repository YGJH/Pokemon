#!/bin/bash
# AlphaZero-style MCTS 自我對弈訓練 (Phase 3)
#
# 對已經 IL 訓練好的模型進行 MCTS 蒸餾訓練。
# 用法:
#   ./scripts/run_mcts_train.sh            # 預設 a0
#   ./scripts/run_mcts_train.sh a1         # 訓練 a1
#
# 會自動編譯 Rust MCTS library。訓練中只有通過 league gate
# (對 frozen IL + 所有 past champions win rate ≥ GATE_SCORE)
# 才會存 checkpoint，最多保留 MAX_CHAMPIONS 個。

set -euo pipefail

# ── Repo root ────────────────────────────────────────────────────────────
cd "$(cd "$(dirname "$0")/.." && pwd)"
# ── 牌組選擇 ─────────────────────────────────────────────────────────────
ARCH="${1:-a0}"

# ── 基本路徑 (所有路徑相對於 python/) ───────────────────────────────────
DATA_DIR="data"
OUT_DIR="checkpoints_${ARCH}_mcts"
IL_CKPT="checkpoints_${ARCH}/ckpt-best.pt"

# ── IL 訓練（先跑 IL 再 MCTS；設 TRAIN_IL=1 啟用）─────────────────────────
TRAIN_IL=0                  # 預設關閉，直接使用現有 checkpoint
D_MODEL=256
LAYERS=10
HEADS=8
FF=1024
DROPOUT=0.1
IL_BATCH_SIZE=512             # 大模型需降 batch（10層 × 512 ≈ 原 4層 × 2048）
IL_EPOCHS=200
IL_PEAK_LR=0.0003
IL_WARMUP=1000
IL_MIN_LR=0.00003
IL_WEIGHT_DECAY=0.01

# ── IL 訓練 ────────────────────────────────────────────────────────────────
if [[ "$TRAIN_IL" == "1" ]]; then
    cd python/
    IL_OUT="${OUT_DIR}_il"
    echo "=== Training IL from scratch: ${ARCH} ==="
    echo "  d_model=$D_MODEL  layers=$LAYERS  heads=$HEADS  ff=$FF"
    uv run python -m ptcg_il.cli train \
        --data-dir "$DATA_DIR" \
        --out-dir "$IL_OUT" \
        --archetype-self 0 \
        --d-model "$D_MODEL" \
        --layers "$LAYERS" \
        --heads "$HEADS" \
        --ff "$FF" \
        --dropout "$DROPOUT" \
        --batch-size "$IL_BATCH_SIZE" \
        --epochs "$IL_EPOCHS" \
        --peak-lr "$IL_PEAK_LR" \
        --warmup "$IL_WARMUP" \
        --min-lr "$IL_MIN_LR" \
        --weight-decay "$IL_WEIGHT_DECAY"
    IL_CKPT="${IL_OUT}/ckpt-best.pt"
    echo "  IL done: $IL_CKPT"
    cd ..

fi

if [[ ! -f "python/$IL_CKPT" ]]; then
    echo "找不到 python/$IL_CKPT" >&2
    echo "現有的 checkpoint:" >&2
    ls -d python/checkpoints* 2>/dev/null | sed 's/^/  /' >&2
    exit 1
fi

# ── MCTS 搜尋參數 ────────────────────────────────────────────────────────
ALL_ARCHETYPES=1            # 1=全部 179 牌組, 0=先用 6 個 𝒟_opp 測試
N_WORKERS=16                # RolloutPool 並行遊戲數 (libcg 實例)
FORWARD_BATCH=4096            # GPU forward batch size
MCTS_ITERATIONS=64          # PUCT iterations per tree
MCTS_C_PUCT=2.0             # PUCT exploration constant
MCTS_K_DET=4                # Determinizations per root (已知對手牌組,不需要 K=8)
MCTS_LEAF_BATCH=4096         # MCTS GPU forward batch size
MCTS_RHO=1.0               # ρ: 多少比例的決策點要跑 MCTS
MCTS_N_ENGINES=20            # MCTS libcg 並行數

# ── Replay buffer ────────────────────────────────────────────────────────
BUFFER_CAPACITY=100000      # 最大容量 (決策點數) — ~1.2 GB
MIN_BUFFER=1000            # 開始訓練的最小 buffer 量

# ── 訓練參數 ─────────────────────────────────────────────────────────────
TOTAL_GAMES=500000           # 總自我對弈場數
GAMES_PER_ITER=1000           # 每輪自我對弈場數
TRAIN_STEPS_PER_ITER=200    # 每輪訓練步數
BATCH_SIZE=1024
LR=0.00002                   # 學習率 (fine-tuning, 低於 IL 的 3e-4)
C_VALUE=0.5                 # Value loss 權重
C_PI=1.0                    # Policy CE loss 權重
# 30, not 0.5.  Measured over a real a1 run, raw grad norms ranged 4.87–34.93
# (median 12.68), so a 0.5 clip bound on **100%** of steps and rescaled each one
# by a different factor between 10x and 70x.  That does not merely discard
# gradient magnitude, it inverts it: every step leaves the clip at norm 0.5, so
# the batch carrying 7x more signal was divided 7x harder.  |g|=15 is also not
# large for this model — at 14.4M params it is a per-parameter RMS of 4e-3,
# while the clipped 0.5 is 1.3e-4.  Step size does not grow from this change:
# AdamW is scale-invariant in steady state (m and sqrt(v) both scale with the
# gradient), so a constant clip is nearly a no-op and `lr` still sets the step.
# Tune against train/grad_clip_frac — aim for 0.05–0.10, not 1.0.
GRAD_CLIP=30

# ── League 評估 ───────────────────────────────────────────────────────────
EVAL_GAMES=100              # 每個對手的對戰場數
EVAL_EVERY_GAMES=100        # 每多少場自我對弈評估一次
GATE_SCORE=0.70             # 對所有對手的最低勝率門檻
MAX_CHAMPIONS=8             # 最多保留幾個 champion（超過則淘汰最低 ELO）

# ── 其他 ─────────────────────────────────────────────────────────────────
SEED=0
DEVICE="${DEVICE:-cuda}"

# ── W&B logging ──────────────────────────────────────────────────────────
WANDB_ENABLED=1             # 1=啟用, 0=關閉
WANDB_PROJECT="pokemon-tcg-mcts"
WANDB_ENTITY="poken"

# ── 編譯 Rust MCTS library ──────────────────────────────────────────────
RUST_DIR="python/ptcg_search"
if command -v cargo &>/dev/null && [[ -f "$RUST_DIR/Cargo.toml" ]]; then
    echo "=== Building libptcg_search.so ==="
    (cd "$RUST_DIR" && cargo build --release 2>&1) || {
        echo "WARNING: cargo build failed — MCTS will use greedy fallback"
    }
else
    echo "WARNING: cargo not found — MCTS will use greedy fallback"
fi

# ── 執行訓練 ─────────────────────────────────────────────────────────────
echo "=== MCTS training: ${ARCH} ==="
echo "IL checkpoint : $IL_CKPT"
echo "Output        : $OUT_DIR"
echo "Total games   : $TOTAL_GAMES"
echo "MCTS iter     : $MCTS_ITERATIONS"
echo "Buffer cap    : $BUFFER_CAPACITY"
echo "Gate score    : $GATE_SCORE"
echo ""

cd python
exec uv run python -m ptcg_rl.mcts_train \
    --il-ckpt "$IL_CKPT" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --iterations "$MCTS_ITERATIONS" \
    --c-puct "$MCTS_C_PUCT" \
    --k-determinizations "$MCTS_K_DET" \
    --n-workers "$N_WORKERS" \
    --forward-batch "$FORWARD_BATCH" \
    --leaf-batch "$MCTS_LEAF_BATCH" \
    --rho "$MCTS_RHO" \
    --mcts-distill \
    --n-engines "$MCTS_N_ENGINES" \
    --buffer-capacity "$BUFFER_CAPACITY" \
    --min-buffer "$MIN_BUFFER" \
    --total-games "$TOTAL_GAMES" \
    --games-per-iter "$GAMES_PER_ITER" \
    --train-steps-per-iter "$TRAIN_STEPS_PER_ITER" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --c-value "$C_VALUE" \
    --c-pi "$C_PI" \
    --grad-clip "$GRAD_CLIP" \
    $([ "$ALL_ARCHETYPES" = "1" ] && echo "--all-archetypes") \
    --eval-games "$EVAL_GAMES" \
    --eval-every-games "$EVAL_EVERY_GAMES" \
    --gate-score "$GATE_SCORE" \
    --max-champions "$MAX_CHAMPIONS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    $([ "$WANDB_ENABLED" = "1" ] && echo "--wandb" || echo "--no-wandb") \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY"
