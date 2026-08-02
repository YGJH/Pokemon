#!/usr/bin/bash
#
# Pre-train + fine-tune, stage 5 only: MCTS fine-tuning of the generalist on
# one deck, gated against that deck's specialist.
#
# Repointed from archetype 0 to 1.  Archetype ids are cluster indices and get
# reassigned every time mining is re-run — this corpus's self_ids are
# [1, 17, 25, 16, 36, 21], so the old `--deck-archetype 0` named a deck that
# does not exist here.  Re-derive rather than trusting this file after any
# re-mine:
#     uv run python -m ptcg_il.cli archetypes --data-dir data --describe
#
# --init-ckpt and --il-ckpt are deliberately different files.  θ starts from
# the generalist (trained on all 70,375 rows); the specialist stays the KL
# anchor and the IL_baseline league opponent, so the gate asks "did fine-tuning
# beat the dedicated specialist?" — a question that cannot be asked if both
# point at the same checkpoint.
#
# Both are produced by:
#     ./scripts/run_pipeline.sh --skip-download --pretrain-finetune --rl-archetype 1

set -euo pipefail

PY_DIR=/home/charles/Documents/Pokemon/python
ARCH=1
GENERALIST_CKPT="$PY_DIR/checkpoints_generalist/ckpt-best.pt"
SPECIALIST_CKPT="$PY_DIR/checkpoints_a${ARCH}/ckpt-best.pt"

cd "$PY_DIR"

for ckpt in "$GENERALIST_CKPT" "$SPECIALIST_CKPT"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "[錯誤] 找不到 $ckpt" >&2
        echo "       先跑 ./scripts/run_pipeline.sh --skip-download --pretrain-finetune --rl-archetype $ARCH" >&2
        exit 1
    fi
done

for((;;)) do

uv run python -m ptcg_rl.mcts_train \
    --data-dir "$PY_DIR/data" \
    --il-ckpt "$SPECIALIST_CKPT" \
    --init-ckpt "$GENERALIST_CKPT" \
    --out-dir "$PY_DIR/checkpoints_a${ARCH}_mcts" \
    --deck-archetype $ARCH \
    --iterations 16 \
    --c-puct 2.0 \
    --k-determinizations 4 \
    --leaf-batch 128 \
    --rho 1.0 \
    --mcts-distill \
    --n-engines 20 \
    --n-workers 16 \
    --forward-batch 1024 \
    --buffer-capacity 10000 \
    --min-buffer 1000 \
    --total-games 5000000 \
    --games-per-iter 200 \
    --train-steps-per-iter 200 \
    --batch-size 1024 \
    --lr 0.00002 \
    --c-value 0.5 \
    --c-pi 1.0 \
    --grad-clip 30 \
    --all-archetypes \
    --eval-games 100 \
    --eval-every-games 100 \
    --gate-score 0.70 \
    --max-champions 8 \
    --seed 0 \
    --device cuda \
    --wandb --wandb-project pokemon-tcg-mcts --wandb-entity poken \
    --wandb-name "pokemon-tcg-mcts-a${ARCH}"

done
