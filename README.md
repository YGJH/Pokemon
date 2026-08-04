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
    --archetype-self 1 --d-model 256 --layers 6 --heads 8 --ff 1024 \
    --dropout 0.0 --batch-size 256 --total-steps 20000
