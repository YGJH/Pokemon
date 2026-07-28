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
IL_CKPT="checkpoints_${ARCH}/ckpt-best.pt"        # 相對於 python/

if [[ ! -f "python/$IL_CKPT" ]]; then
    echo "找不到 python/$IL_CKPT" >&2
    echo "現有的 checkpoint:" >&2
    ls -d python/checkpoints* 2>/dev/null | sed 's/^/  /' >&2
    exit 1
fi

# ── 基本路徑 (所有路徑相對於 python/) ───────────────────────────────────
DATA_DIR="data"
OUT_DIR="checkpoints_${ARCH}_mcts"

# ── MCTS 搜尋參數 ────────────────────────────────────────────────────────
ALL_ARCHETYPES=1            # 1=全部 179 牌組, 0=只用 6 個 𝒟_opp
MCTS_ITERATIONS=64          # PUCT iterations per tree (推論用較低)
MCTS_C_PUCT=2.0             # PUCT exploration constant
MCTS_K_DET=4                # Determinizations per root (已知對手牌組,不需要 K=8)
MCTS_LEAF_BATCH=512         # GPU forward batch size
MCTS_RHO=0.05               # ρ: 多少比例的決策點要跑 MCTS
MCTS_N_ENGINES=4            # libcg 並行數

# ── Replay buffer ────────────────────────────────────────────────────────
BUFFER_CAPACITY=100000      # 最大容量 (決策點數)
MIN_BUFFER=10000            # 開始訓練的最小 buffer 量

# ── 訓練參數 ─────────────────────────────────────────────────────────────
TOTAL_GAMES=5000            # 總自我對弈場數
GAMES_PER_ITER=50           # 每輪自我對弈場數
TRAIN_STEPS_PER_ITER=200    # 每輪訓練步數
BATCH_SIZE=256
LR=0.0001                   # 學習率 (fine-tuning, 低於 IL 的 3e-4)
C_VALUE=0.5                 # Value loss 權重
C_PI=1.0                    # Policy CE loss 權重
GRAD_CLIP=1.0

# ── League 評估 ───────────────────────────────────────────────────────────
EVAL_GAMES=100              # 每個對手的對戰場數
EVAL_EVERY_GAMES=250        # 每多少場自我對弈評估一次
GATE_SCORE=0.70             # 對所有對手的最低勝率門檻
MAX_CHAMPIONS=10            # 最多保留幾個 champion

# ── 其他 ─────────────────────────────────────────────────────────────────
SEED=0
DEVICE="${DEVICE:-cuda}"    # 可透過環境變數覆蓋

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
    --leaf-batch "$MCTS_LEAF_BATCH" \
    --rho "$MCTS_RHO" \
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
    $([ "$ALL_ARCHETYPES" = "1" ] && echo "--all-archetypes" || echo "--no-all-archetypes") \
    --eval-games "$EVAL_GAMES" \
    --eval-every-games "$EVAL_EVERY_GAMES" \
    --gate-score "$GATE_SCORE" \
    --max-champions "$MAX_CHAMPIONS" \
    --seed "$SEED" \
    --device "$DEVICE"
