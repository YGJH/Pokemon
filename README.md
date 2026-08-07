Running it

cd python
uv run python -m ptcg_il.cli archetypes --data-dir data --describe

Three modes, and they answer different questions:

# The table — for a human deciding what to train
uv run python -m ptcg_il.cli archetypes --data-dir data --describe

# Bare ids, best-supported first — this is what run_pipeline.sh consumes
uv run python -m ptcg_il.cli archetypes --data-dir data --top 2      # → "1 25"

# Ask for more than qualify and it tells you, rather than padding
uv run python -m ptcg_il.cli archetypes --data-dir data --top 6
# WARNING Only 4 of the requested 6 archetypes have enough held-out data


# Everything, pinned to a1 only, no RL stage
./scripts/run_pipeline.sh --n-days 5 --target-episodes 2000 --archetypes 1 --no-rl

That runs download → mine → build-shards → train, and lands in python/checkpoints_a1/. Or by hand, from python/:

cd python

# 1. Download + mine. Archetype ids are seeded from data/archetypes.json by
#    default now, so a1 stays a1.
uv run python -m ptcg_mine.mine --n-days 5 --target-episodes 2000 \
    --raw-dir raw --out-dir data

# 2. Shards. Rebuilds itself because its fingerprint covers archetypes.json's
#    content, which the new corpus changes. ~35 min.
uv run python -m ptcg_il.cli build-shards --raw-dir raw --out-dir data

# 3. Train a1
uv run python -m ptcg_il.cli train --data-dir data --out-dir checkpoints_a1 \
    --archetype-self 1 --d-model 256 --layers 14 --heads 8 --ff 1024 \
    --dropout 0.2 --batch-size 256 --total-steps 2000000

uv run python -m ptcg_mine.mine --skip-download
    --raw-dir raw --out-dir data

  新标志：--ensemble，后跟一个 glob 或路径列表。用法：
  ./scripts/build_submit.sh a1 --ensemble "python/checkpoints_a1_s*/ckpt-best.pt"
  这也接受显式路径：
  ./scripts/build_submit.sh a1 --ensemble \
      python/checkpoints_a1_s0/ckpt-best.pt \
      python/checkpoints_a1_s1/ckpt-best.pt


There's no separate "ensemble trainer" — you train N ordinary specialists with different --seeds, then combine them at eval/packaging time.

1. Get the real archetype id (never hardcode it):
cd python && uv run python -m ptcg_il.cli archetypes --data-dir data --describe

2. Train the members — scripts/run_ensemble_train.sh <ARCH> <N> [extra train flags]:
./scripts/run_ensemble_train.sh 1 3 --total-steps 30000 --batch-size 512
Sequential on one GPU. Member S gets --seed S and lands in python/checkpoints_a1_s<S>/. Extra flags are forwarded verbatim to every ptcg_il.cli train call.

Equivalent by hand:
cd python
uv run python -m ptcg_il.cli train --data-dir data --out-dir checkpoints_a1_s0 \
    --archetype-self 1 --seed 0

3. Measure it — --eval-only with 2+ --ckpt flags dispatches to the ensemble path (cli.py:530), which evals each member and the ensemble and prints a comparison table:
cd python
uv run python -m ptcg_il.cli train --eval-only --data-dir data \
    --archetype-self 1 --eval-split test \
    --ckpt checkpoints_a1_s0/ckpt-best.pt \
    --ckpt checkpoints_a1_s1/ckpt-best.pt \
    --ckpt checkpoints_a1_s2/ckpt-best.pt \
    --record-baseline          # writes an "ens-N" record SHA-pinned to all members
# add --live-eval for head-to-head vs the best single member (ship rule: Wilson LB > 50%)

4. Package:
./scripts/build_submit.sh --ensemble "python/checkpoints_a1_s*/ckpt-best.pt"

./scripts/build_submit.sh --ensemble "python/checkpoints_a1_s[0-6]/ckpt-best.pt"


Ensemble mode is implicitly --no-mcts (greedy only) and produces submission-greedy-ens3.tar.gz with model_0.pt…model_2.pt + ensemble.json.

Two constraints worth knowing: all members must share the same decklist (a mismatch raises — different vocab/archetype SHAs only warn), and different architectures per member are allowed since the encode path is per-member.