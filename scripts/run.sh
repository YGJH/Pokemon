#!/usr/bin/bash
cd python/
for ((;;)) do
uv run python -m ptcg_mine.mine --n-days 5 --target-episodes 20000000 \
    --raw-dir raw --out-dir data

uv run python -m ptcg_il.cli build-shards --raw-dir raw --out-dir data
sleep 100
done
