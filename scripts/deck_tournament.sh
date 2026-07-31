#!/bin/bash
# Deck tournament — all checkpoints play each other with MCTS, ranked by ELO.
#
#   ./scripts/deck_tournament.sh              # all checkpoints_*
#   ./scripts/deck_tournament.sh a0 a1 a2     # specific decks
#   GAMES=100 ./scripts/deck_tournament.sh    # more games per pair

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

GAMES="${GAMES:-40}"
ITERATIONS="${ITERATIONS:-64}"

# Build Rust
if command -v cargo &>/dev/null; then
    (cd python/ptcg_search && cargo build --release 2>&1) || true
fi

cd python
exec uv run python -m ptcg_rl.tournament \
    --data-dir data \
    --games "$GAMES" \
    --iterations "$ITERATIONS" \
    ${1:+--decks "$@"}
