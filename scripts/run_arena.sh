#!/bin/bash
# Cross-deck arena — 所有 checkpoint 互相對打，決出 model 強度與 deck 強度。
#
# 每個 entry 都帶著自己 .pt 裡記錄的牌組進場（elo_calibrate.sh 不會，
# 那邊兩邊永遠用同一組牌，跨牌組比較因此無效）。
#
# 評分模型: R_i = S_model(i) + D_deck(i)，generalist/best 定錨 1500、ΣD = 0。
# S 和 D 之所以能分開，是因為 generalist 會以同一份權重打過每一副牌組。
#
# 用法:
#   ./scripts/run_arena.sh              # 100 games/pair，完整名單
#   ./scripts/run_arena.sh 20           # 20 games/pair
#   ./scripts/run_arena.sh 100 --skip-steps   # 只留 best/last + 最新 champion
#   ./scripts/run_arena.sh 0            # dry-run: 只印名單與成本估算
#
# 中斷後續跑: ./scripts/run_arena.sh 100 --resume
# 只重算評分（不打任何一場）: cd python && uv run python -m ptcg_rl.arena --fit-only

set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

GAMES="${1:-100}"
shift || true

DATA_DIR="data"
OUT="arena_ratings.json"

if [[ ! -d python/data ]]; then
    echo "找不到 python/$DATA_DIR — 先跑 mining (./scripts/run_pipeline.sh)" >&2
    exit 1
fi

cd python

if [[ "$GAMES" == "0" ]]; then
    exec uv run python -m ptcg_rl.arena \
        --data-dir "$DATA_DIR" --out "$OUT" --dry-run "$@"
fi

echo "=== Cross-deck arena ==="
echo "Games/pair : $GAMES"
echo "Output     : python/$OUT"
echo ""

exec uv run python -m ptcg_rl.arena \
    --data-dir "$DATA_DIR" \
    --out "$OUT" \
    --games "$GAMES" \
    "$@"
