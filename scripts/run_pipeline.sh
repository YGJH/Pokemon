#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — 完整训练管线 (Phase 1 → Phase 4 → Training)
#
# 用法:
#   ./scripts/run_pipeline.sh                        # 全部执行（下载 + 挖掘 + 建分片 + 训练）
#   ./scripts/run_pipeline.sh --skip-download          # 跳过下载（已有 raw/ 资料）
#   ./scripts/run_pipeline.sh --skip-download --no-train  # 只做到建分片，不训练
#
# 环境要求:
#   - uv (Python 3.11+)
#   - ~/.kaggle/kaggle.json (下载需要)
# =============================================================================

set -euo pipefail

# ── 预设参数 ────────────────────────────────────────────────────────────────
SKIP_DOWNLOAD=""
NO_TRAIN=""
SKIP_RUST=""
# 相对路径一律以 python/ 为基准（管线全程在 python/ 下执行，
# 因为 ptcg_mine / ptcg_il 套件位于该目录，必须是 cwd 才 import 得到）。
RAW_DIR="raw"
DATA_DIR="data"
CHECKPOINT_DIR="checkpoints"
N_DAYS=20
TARGET_EPISODES=10000
SEED=0
K_EXPERTS=10
G_MIN=50
JACCARD_THRESH=0.90
BATCH_SIZE=1024
EPOCHS=10
# echo $1
# ── 解析参数 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download)
            SKIP_DOWNLOAD="--skip-download"
            shift
            ;;
        --no-train)
            NO_TRAIN="true"
            shift
            ;;
        --skip-rust)
            SKIP_RUST="true"
            shift
            ;;
        --raw-dir)
            RAW_DIR="$2"; shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"; shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"; shift 2
            ;;
        --n-days)
            N_DAYS="$2"; shift 2
            ;;
        --target-episodes)
            TARGET_EPISODES="$2"; shift 2
            ;;
        --seed)
            SEED="$2"; shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2
            ;;
        --epochs)
            EPOCHS="$2"; shift 2
            ;;
        --resume)
            RESUME_CKPT="--resume $2"; shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "管线步骤: Rust engine → Phase 1-2(mine) → Phase 3(build-shards) → QA → Training"
            echo ""
            echo "路径说明: 相对路径以 python/ 为基准（预设 = python/raw, python/data,"
            echo "          python/checkpoints）。要指定别处请给绝对路径。"
            echo ""
            echo "选项:"
            echo "  --skip-download        跳过 Phase 1 下载（raw/ 已有资料时使用）"
            echo "  --skip-rust            跳过 Rust search engine 编译"
            echo "  --no-train             只做 mine + build-shards，不训练"
            echo "  --raw-dir DIR          raw 目录 (默认: python/raw)"
            echo "  --data-dir DIR         产出目录 (默认: python/data)"
            echo "  --checkpoint-dir DIR   checkpoint 目录 (默认: python/checkpoints)"
            echo "  --n-days N             下载的天数 (默认: 20)"
            echo "  --target-episodes N    总 episode 目标 (默认: 10000)"
            echo "  --seed N               随机种子 (默认: 0)"
            echo "  --batch-size N         训练 batch size (默认: 2048)"
            echo "  --epochs N             训练 epoch 数 (默认: 10)"
            echo "  --resume PATH          从指定 checkpoint 恢复训练"
            echo "  --help, -h             显示此帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"; exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PY_DIR="$PROJECT_DIR/python"
RUST_DIR="$PY_DIR/ptcg_search"

# 相对路径 → 绝对路径（以 python/ 为基准），避免各步骤 cwd 不同时指到不同目录
abspath() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *)  printf '%s' "$PY_DIR/$1" ;;
    esac
}
RAW_DIR="$(abspath "$RAW_DIR")"
DATA_DIR="$(abspath "$DATA_DIR")"
CHECKPOINT_DIR="$(abspath "$CHECKPOINT_DIR")"

cd "$PY_DIR"

echo "============================================"
echo " Pokémon TCG IL — 完整训练管线"
echo "============================================"
echo "专案目录: $PROJECT_DIR"
echo "执行目录: $PY_DIR"
echo "Raw 目录: $RAW_DIR"
echo "资料目录: $DATA_DIR"
echo "Checkpoint 目录: $CHECKPOINT_DIR"
echo ""

# ── 步骤 1: 编译 Rust search engine ──────────────────────────────────────
echo "══════════════════════════════════════════════"
echo " 步骤 1/4: 编译 Rust search engine"
echo "══════════════════════════════════════════════"

RUST_LIB="$RUST_DIR/target/release/libptcg_search.so"
if [[ "$SKIP_RUST" == "true" ]]; then
    echo "--skip-rust 已指定，跳过编译。"
elif ! command -v cargo >/dev/null 2>&1; then
    echo "[警告] 找不到 cargo — 跳过 Rust engine 编译。" >&2
    echo "       只影响 live-eval 的 search planner 对手；mine / 训练不受影响。" >&2
    echo "       安装: https://rustup.rs  或用 --skip-rust 静音此警告。" >&2
else
    echo "\$ cargo build --release --manifest-path $RUST_DIR/Cargo.toml"
    cargo build --release --manifest-path "$RUST_DIR/Cargo.toml"
    if [[ ! -f "$RUST_LIB" ]]; then
        echo "[错误] cargo 编译成功但未产生 $RUST_LIB" >&2
        exit 1
    fi
    echo "✓ Rust engine 编译完成 — $RUST_LIB"
fi
echo ""

# ── 步骤 2: Phase 0–2 语料挖掘 ────────────────────────────────────────────
echo "══════════════════════════════════════════════"
echo " 步骤 2/4: 语料挖掘 (Phase 0–2)"
echo "══════════════════════════════════════════════"
MINE_CMD="uv run python -m ptcg_mine.mine \
    --raw-dir $RAW_DIR \
    --out-dir $DATA_DIR \
    --n-days $N_DAYS \
    --target-episodes $TARGET_EPISODES \
    --seed $SEED \
    --k-experts $K_EXPERTS \
    --g-min $G_MIN \
    --jaccard-thresh $JACCARD_THRESH \
    $SKIP_DOWNLOAD"

if [[ -z "$SKIP_DOWNLOAD" ]]; then
    QUOTA=$(( TARGET_EPISODES / N_DAYS ))
    echo "[提示] 本次下载约需 $(( N_DAYS * 100 + QUOTA * N_DAYS )) 次 Kaggle API 呼叫"
    echo "       (${N_DAYS} 天 x 每天 ${QUOTA} 个 episode)。数量过大会触发 HTTP 429 限流；"
    echo "       下载可续传，建议先用较小的 --n-days / --target-episodes 分批累积。"
    echo ""
fi

echo "\$ $MINE_CMD"
echo ""
# 不让 set -e 直接静默中断：下载失败(exit 2)要给出可操作的说明
set +e
eval "$MINE_CMD"
MINE_RC=$?
set -e

if [[ $MINE_RC -eq 2 ]]; then
    echo "" >&2
    echo "[错误] Phase 1 下载失败（通常是 Kaggle HTTP 429 限流）。" >&2
    echo "       已下载的档案会保留在 $RAW_DIR/，下次执行会自动续传。" >&2
    echo "       建议: 等 15-60 分钟后，用较小的批次重跑，例如" >&2
    echo "         $0 --n-days 5 --target-episodes 500" >&2
    echo "       或先用现有资料继续: $0 --skip-download" >&2
    exit 2
elif [[ $MINE_RC -ne 0 ]]; then
    echo "" >&2
    echo "[错误] 语料挖掘失败 (exit $MINE_RC) — 检查上方日志" >&2
    exit $MINE_RC
fi

# 检查必要产出
if [[ ! -f "$DATA_DIR/vocab.json" ]]; then
    echo "[错误] mine 完成但未产生 $DATA_DIR/vocab.json — 检查上方日志" >&2
    exit 1
fi
echo ""
echo "✓ 语料挖掘完成 — vocab.json, archetypes.json 已产生"
echo ""

# ── 步骤 3: Phase 3 特征化 ────────────────────────────────────────────────
echo "══════════════════════════════════════════════"
echo " 步骤 3/4: 建立训练分片 (Phase 3)"
echo "══════════════════════════════════════════════"

BS_CMD="uv run python -m ptcg_il.cli build-shards \
    --raw-dir $RAW_DIR \
    --out-dir $DATA_DIR \
    --k-experts $K_EXPERTS \
    --g-min $G_MIN \
    --jaccard-thresh $JACCARD_THRESH"

echo "\$ $BS_CMD"
echo ""
eval "$BS_CMD"

if [[ ! -f "$DATA_DIR/meta.parquet" ]]; then
    echo "[错误] build-shards 完成但未产生 $DATA_DIR/meta.parquet" >&2
    exit 1
fi

SHARD_COUNT=$(ls "$DATA_DIR/shards/"*.npz 2>/dev/null | wc -l)
echo ""
echo "✓ 分片建立完成 — $SHARD_COUNT 个 .npz 文件, meta.parquet 已产生"
echo ""

# ── 步骤 4: 训练 ──────────────────────────────────────────────────────────
if [[ "$NO_TRAIN" == "true" ]]; then
    echo "═══ --no-train 已指定，跳过训练 ═══"
    echo "管线结束。产物在: $DATA_DIR/"
    exit 0
fi

echo "══════════════════════════════════════════════"
echo " 步骤 4/4: 训练 IL Policy"
echo "══════════════════════════════════════════════"

# 前置检查：训练需要 torch，缺了就在这里明确失败（而不是 import 时炸掉）
if ! uv run python -c "import torch" >/dev/null 2>&1; then
    echo "[错误] 找不到 torch — 无法训练。" >&2
    echo "       执行 'uv sync' 安装（pyproject.toml 已设定 CUDA 12.8 wheel）。" >&2
    echo "       分片已建立完成，之后可直接跑:" >&2
    echo "         cd $PY_DIR && uv run python -m ptcg_il.cli train --data-dir $DATA_DIR --out-dir $CHECKPOINT_DIR" >&2
    exit 1
fi

TRAIN_CMD="uv run python -m ptcg_il.cli train \
    --data-dir $DATA_DIR \
    --out-dir $CHECKPOINT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS"

if [[ -n "${RESUME_CKPT:-}" ]]; then
    TRAIN_CMD="$TRAIN_CMD $RESUME_CKPT"
fi

echo "\$ $TRAIN_CMD"
echo ""
eval "$TRAIN_CMD"

echo ""
echo "============================================"
echo " 管线完成！"
echo "============================================"
echo "最佳模型: $CHECKPOINT_DIR/ckpt-best.pt"
echo "训练产出: $CHECKPOINT_DIR/"
echo "语料产物: $DATA_DIR/"
echo ""
echo "下一步 (需先 cd $PY_DIR):"
echo "  # 评估模型"
echo "  uv run python -m ptcg_il.cli train --eval-only --resume $CHECKPOINT_DIR/ckpt-best.pt"
echo ""
echo "  # 用 live engine 对战评估"
echo "  uv run python -m ptcg_il.cli train --eval-only --live-eval --resume $CHECKPOINT_DIR/ckpt-best.pt"
