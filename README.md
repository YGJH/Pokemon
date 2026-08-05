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



  新标志：--ensemble，后跟一个 glob 或路径列表。用法：
  ./scripts/build_submit.sh a1 --ensemble "python/checkpoints_a1_s*/ckpt-best.pt"
  这也接受显式路径：
  ./scripts/build_submit.sh a1 --ensemble \
      python/checkpoints_a1_s0/ckpt-best.pt \
      python/checkpoints_a1_s1/ckpt-best.pt



Design below. Sizes: your ckpt-best.pt is 119 MB (optimizer state included); packaging strips it to EMA weights — the current greedy bundle is 40 MB total, so 3 members lands around 90–100 MB. Worth confirming against the competition's bundle limit before N grows.

---
A. --seed in the IL trainer

--seed INT (default 42) on ptcg_il.cli train: seeds torch.manual_seed before model construction and replaces the hardcoded seed=42 at train/loop.py:570, and gets stamped into Policy.config so a member is identifiable later.

Safe because the train/val/test split comes from shard filename prefixes (dataset.py:266), not from the RNG — varying the seed changes batch order and init, never the split. So member val/test numbers stay comparable.

One caveat I won't fix unless you want it: bit-exact GPU reproducibility also needs torch.use_deterministic_algorithms(True) + cuDNN flags, which cost throughput and lack kernels for some ops. Without them --seed 0 gives you a distinct, near-reproducible member, not a bit-identical one.

Separately: with --dropout 0.0, init and batch order are the only decorrelation. Dropout > 0 would add more, but it changes the config you've already tuned — flagging it, not changing it.

B. python/ptcg_il/ensemble.py — EnsemblePolicy

An nn.Module over an nn.ModuleList of members.

EnsemblePolicy.from_checkpoints(paths, all_card_feat, device) reads each checkpoint's own config, so heterogeneous members work for free if you later want them, and asserts every member shares deck.vocab_sha1 / archetypes_sha1 / decklist. Without that assert you can ensemble models trained against different vocabularies and get a bundle that runs fine and loses almost everything — the exact silent failure CLAUDE.md documents for single checkpoints.

- forward(x, history_h=None) -> (logits, value, hist) — per member: masked_fill(~opt_mask, -1e9) → softmax → mean → return log(mean_p). Log rather than raw probability so it stays drop-in at call sites that do .masked_fill(...).argmax(); argmax is unchanged by the log. Value = mean of member values (only MCTS reads it).
- select_multi(x, history_h=None) — the ensemble AR loop below.

C. AR loop, in model/policy.py

select_multi(policy, x, history_h) gains a one-line dispatch at the top: if the object defines its own select_multi, delegate. Policy doesn't define one, so there's no recursion, and every existing call site works unchanged — live_eval.PolicyAgent:174, the greedy main.py template, arena.

_select_multi_raw generalizes to N members: N msgru states, and per step each member produces (logits_i, o_i). STOP-masking and picked-masking apply to each member's logits before its softmax, then probabilities average and one argmax decides. Each member advances its own msgru with its own o_i at the ensemble-chosen index — members condition on the joint decision, which is what makes this an ensemble rather than N independent agents.

Regression test: a one-member list must reproduce today's _select_multi_raw output exactly, on a fixture with a real multi-select and a STOP column.

D. Packaging

build_submission.py: --ckpt becomes action="append". One path → today's data/model.pt and today's main.py, byte-identical. Two or more → data/model_0.pt … model_{N-1}.pt + an ensemble.json manifest, and a greedy template that constructs EnsemblePolicy. The vocab/archetypes SHA pin check runs per member. ensemble.py joins MODEL_FILES for the import rewrite.

build_submit.sh a1 --no-mcts --ensemble globs python/checkpoints_a1_s*/ckpt-best.pt; repeated explicit paths also work. Output is submission-greedy-ens3.tar.gz — a distinct name for the same reason --no-mcts already has one: three identical-looking bundles that overwrite each other is how you lose track of what you uploaded.

E. Measurement — the actual deliverable

The bundle is not evidence. Two additions:

1. ptcg_il.cli train --eval-only --ensemble-ckpt A --ensemble-ckpt B … → held-out test top-1/top-3 for the ensemble and each member.
2. Live eval / arena: PolicyAgent accepts anything with forward + select_multi, so an EnsemblePolicy drops in. Run ensemble vs. best single member head-to-head.

Ship rule: ensemble wins only if its live win-rate against the best single member has a Wilson lower bound above 50% (wilson_interval exists at live_eval.py:82). Offline top-1 decides nothing here — CLAUDE.md already records a 78%-offline / 48%-live gap on this project. Then add a 4th and 5th member and stop when the live curve flattens.

Cost is N× forward passes per decision, on CPU during live eval.

The training pipeline you asked for

cd python
for S in 0 1 2; do
  uv run python -m ptcg_il.cli train --data-dir data --out-dir checkpoints_a1_s$S \
      --archetype-self 1 --seed $S \
      --d-model 256 --layers 6 --heads 8 --ff 1024 \
      --dropout 0.0 --batch-size 256 --total-steps 20000
done

Sequential, one GPU. I'd wrap it as scripts/run_ensemble_train.sh a1 3.

Build order: A (seed) → C (AR refactor + 1-member regression test) → B (EnsemblePolicy) → E (measure on your existing checkpoints_a1 alone, to prove the plumbing is neutral) → train the 3 seeds → measure → D (package) only if E says it won.