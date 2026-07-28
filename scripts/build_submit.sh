#!/bin/bash
# 打包 Kaggle 提交檔。
#
# checkpoint 一定要是「用現在這份 python/data 訓練出來的」那一個。
# archetype id 只是分群索引，重跑 mine 就會重新編號，所以 checkpoints_a<N>/
# 的 N 不代表任何固定的牌組 —— 舊語料留下來的目錄看起來完全正常，卻是對著
# 另一份 vocab 訓練的。build_submission.py 現在會比對 checkpoint 裡釘住的
# vocab/archetypes sha1，對不上就中止，而不是打包出一個「跑得完但幾乎全輸」
# 的 bundle。
#
# 要打包哪一副牌用第一個參數指定，預設 a0（資料量最大的那副）:
#   ./scripts/build_submit.sh          # checkpoints_a0
#   ./scripts/build_submit.sh a1       # checkpoints_a1
#
# 會自動編譯 Rust MCTS library (libptcg_search.so) 並打包進 submission。
# 如果 cargo 找不到，會跳過並在 submission 中使用 greedy fallback。

set -euo pipefail

ARCH="${1:-a0}"
CKPT="python/checkpoints_${ARCH}/ckpt-best.pt"

if [[ ! -f "$CKPT" ]]; then
    echo "找不到 $CKPT" >&2
    echo "現有的 checkpoint:" >&2
    ls -d python/checkpoints* 2>/dev/null | sed 's/^/  /' >&2
    exit 1
fi

# ── Build Rust MCTS library ─────────────────────────────────────────────
RUST_DIR="python/ptcg_search"
if command -v cargo &>/dev/null && [[ -f "$RUST_DIR/Cargo.toml" ]]; then
    echo "Building libptcg_search.so..."
    (cd "$RUST_DIR" && cargo build --release 2>&1) || echo "WARNING: cargo build failed — submission will use greedy fallback"
else
    echo "WARNING: cargo not found — skipping Rust build, MCTS will fall back to greedy policy"
fi
# ─────────────────────────────────────────────────────────────────────────

uv run python scripts/build_submission.py \
    --data-dir python/data \
    --ckpt "$CKPT" \
    --out submission.tar.gz
