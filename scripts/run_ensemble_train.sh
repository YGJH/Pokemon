#!/bin/bash
# Train N independently-seeded specialist models for ensemble.
#
# Usage: ./scripts/run_ensemble_train.sh <ARCH> <N> [--extra-flags ...]
#
#   ./scripts/run_ensemble_train.sh 1 3
#   ./scripts/run_ensemble_train.sh 1 5 --total-steps 30000 --batch-size 512
#
# Trains sequentially on one GPU. Each member gets a different --seed.

set -euo pipefail

ARCH="${1:?Usage: $0 <ARCH> <N> [--extra-flags ...]}"
N="${2:?Usage: $0 <ARCH> <N> [--extra-flags ...]}"
shift 2

# Archetype id: strip 'a' prefix if present
ARCH_ID="${ARCH#a}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../python"

for S in $(seq 0 $((N - 1))); do
    echo "=== Training member $S/$N (seed=$S, archetype=$ARCH_ID) ==="
    uv run python -m ptcg_il.cli train \
        --data-dir data \
        --out-dir "checkpoints_a${ARCH_ID}_s${S}" \
        --archetype-self "$ARCH_ID" \
        --seed "$S" \
        "$@"
    echo "=== Member $S done ==="
done

echo "=== All $N members trained ==="
