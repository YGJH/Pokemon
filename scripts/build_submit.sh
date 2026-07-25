#!/bin/bash

uv run python scripts/build_submission.py \
    --data-dir python/data \
    --ckpt python/checkpoints_a2/ckpt-best.pt \
    --out submission.tar.gz
