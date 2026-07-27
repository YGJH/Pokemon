"""Self-play RL on top of the IL policy (RL_SPEC.md).

Phases implemented here: **R1** (critic repair) and **R2** (PPO with a KL anchor
to the frozen IL policy, one deck, self-play).  MCTS distillation (R3) and the
two-deck league (R4) are deliberately absent — RL_SPEC §11 makes R2 the go/no-go,
and adding search or a league before it passes only makes a failure harder to
attribute.

Nothing here changes what the policy consumes or emits: RL changes how the
policy is *trained*, and ``TRANSFORMER_IL_SPEC.md`` remains authoritative for
tensor layout and the ``agent()`` contract.
"""
