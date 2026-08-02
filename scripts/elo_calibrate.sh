#!/bin/bash
# ELO 校正 — round-robin 模式，所有 checkpoint 互相對打，定錨 ELO。
#
# ckpt-best 固定在 1500，其他模型根據對戰結果計算 ELO。
# 結果寫入 checkpoints_${ARCH}_mcts/elo_ratings.json。
#
# 預設為 round-robin 模式：每對模型打 10 場（Python 端預設）。
# 舊的 reference 模式（只跟 ckpt-best 打）可用 --mode reference 直接呼叫 Python。
#
# 用法:
#   ./scripts/elo_calibrate.sh           # 預設 a0, round-robin, 10 games/pair
#   ./scripts/elo_calibrate.sh a0 30     # 自訂 30 games/pair
#   ./scripts/elo_calibrate.sh a1        # 校正 a1
#
# 會自動跳過沒有 config record 的損壞存檔。

set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

ARCH="${1:-a0}"
GAMES="${2:-}"

DATA_DIR="data"
OUT_DIR="checkpoints_${ARCH}_mcts"
IL_CKPT="checkpoints_${ARCH}"

if [[ ! -d "python/$IL_CKPT" ]]; then
    echo "找不到 python/$IL_CKPT" >&2
    exit 1
fi

echo "=== ELO calibration: ${ARCH} ==="
echo "IL checkpoint : $IL_CKPT"
echo "Output        : $OUT_DIR"
if [[ -n "$GAMES" ]]; then
    echo "Games/pair    : $GAMES"
else
    echo "Games/pair    : (default: 10 round-robin)"
fi
echo ""

cd python
if [[ -n "$GAMES" ]]; then
    exec uv run python -m ptcg_rl.elo_calibrate \
        --il-ckpt "$IL_CKPT" \
        --data-dir "$DATA_DIR" \
        --out-dir "$OUT_DIR" \
        --games "$GAMES"
else
    exec uv run python -m ptcg_rl.elo_calibrate \
        --il-ckpt "$IL_CKPT" \
        --data-dir "$DATA_DIR" \
        --out-dir "$OUT_DIR"
fi
