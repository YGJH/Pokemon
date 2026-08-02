checkpoints/ is a byte-identical duplicate of checkpoints_generalist/ (same size, same mtime, same SHAs) — dropping it.

Now, the problem with the deck term, and the design.

The identifiability problem

You asked for R_i = S_model + D_deck. But every specialist only ever plays its own deck. If a1's models never play anything but the a1 deck, then S_a1-model and D_a1-deck appear only as a sum — no amount of games separates them. The fit would still converge and print two columns, and they'd be arbitrary. That's exactly the kind of number that reads as insight and isn't.

The fix: the generalist is the bridge. checkpoints_generalist/ckpt-best.pt has archetype_self=None and was trained across the whole corpus — it is the one model that can legitimately pilot any deck (it's what mcts_train warm-starts every specialist from). So it enters the arena once per deck. Same weights, six decks: every difference between those entries is pure deck strength, and that pins D for everyone else.

This carries one assumption I want stated out loud rather than buried: the deck effect is additive and model-independent — a1's deck is worth +X to anybody. If a1's specialist exploits its deck better than the generalist can, that surplus lands in S, not D. That's the honest reading, and it's the interesting quantity anyway.

Roster (19 entries, 6 decks)

┌─────────────────────────────────────────────────┬───────┬──────────────────┐
│                      entry                      │ deck  │       role       │
├─────────────────────────────────────────────────┼───────┼──────────────────┤
│ generalist/ckpt-best                            │ FIXED │ anchor, 1500     │
├─────────────────────────────────────────────────┼───────┼──────────────────┤
│ generalist/ckpt-last                            │ FIXED │ competitor       │
├─────────────────────────────────────────────────┼───────┼──────────────────┤
│ generalist/ckpt-best × {a1, a16, a17, a25, a36} │ each  │ 5 bridge entries │
├─────────────────────────────────────────────────┼───────┼──────────────────┤
│ a1/ckpt-best, a1/ckpt-last                      │ a1    │ competitor       │
├─────────────────────────────────────────────────┼───────┼──────────────────┤
│ a17/ckpt-best, a17/ckpt-last                    │ a17   │ competitor       │
├─────────────────────────────────────────────────┼───────┼──────────────────┤
│ a1_mcts/champ-001600, a1_mcts/mcts-last         │ a1    │ competitor       │
├─────────────────────────────────────────────────┼───────┼──────────────────┤
│ a16_mcts/champ-000200, a16_mcts/mcts-last       │ a16   │ competitor       │
├─────────────────────────────────────────────────┼───────┼─────────────────
│ a17_mcts/mcts-last                              │ a17   │ competitor       │
├─────────────────────────────────────────────────┼───────┼─────────────────
│ a25_mcts/mcts-last                              │ a25   │ competitor       │
├─────────────────────────────────────────────────┼───────┼─────────────────
│ a36_mcts/champ-000600, a36_mcts/mcts-last       │ a36   │ competitor       │
└─────────────────────────────────────────────────┴───────┴─────────────────

19 entries → 171 pairs × 100 games ≈ 17.1k games, ~2.1h. Up from my 1.1h estis real work; --games dials it. --include-steps adds the intermediateckpt-step-* and the full a1 champion ladder if you want to see the drift trajectory rated (→35 entries, ~7h).

Modules

python/ptcg_rl/rating.py (new, ~150 lines) — pure numpy/scipy, no torch, no engine.
- fit_bradley_terry(pairs, entries, anchor) -> RatingTable
- Model: P(i beats j) = σ(ln10/400 · (R_i − R_j)), R_i = S_{model(i)} + D_{deck(i)}, constraints R_anchor = 1500, Σ D = 0.
- CIs from the inverse Hessian at the optimum.
- Raises if the comparison graph is disconnected or the design matrix is rank-deficient — the two ways this silently returns garbage.

python/ptcg_rl/arena.py (new, ~350 lines) — roster + match driving.
- Roster built from each .pt's own deck record; a checkpoint without one is
- Refuses to run if vocab_sha1 or archetypes_sha1 differ across the roster. Card ids are vocab indices and archetype ids are cluster indices — mixing generations is silently wrong,
not loudly wrong.
- Per pair: games/2 with entry i in seat 0, games/2 with entry j in seat 0, decks following the model in both halves (RustVecEnv(deck_self=deck_of(seat0), deck_opp=deck_of(seat1),
our_player=0)). Greedy actors, per-game seeds vary the shuffle — matching ho.
- Writes arena_ratings.json (full result matrix + fit, so the fit can be re-run without replaying games) and prints the three-column table.

scripts/run_arena.sh, python/tests/test_arena_rating.py.

Testing

rating.py is the part that can be wrong silently, so it gets the real tests: generate matches from known ratings and deck effects, assert recovery within CI; assert a roster with no
bridge entries raises instead of returning plausible numbers. Per your CLAUDtation — delete the bridge, confirm red; break the anchor constraint, confirm red — and the recovery test fails on zero pairs examined.

arena.py gets a fake-engine test for the deck-follows-model swap, since that's the specific bug in the current code.
