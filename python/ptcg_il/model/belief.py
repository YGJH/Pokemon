"""Game-log constants (the BeliefModule GRU encoder has been removed).

The ``L_LOG_MAX`` and ``LOG_FEAT_DIM`` constants are still referenced by the
featurizer and belief-label code (until Task 13 removes those as well).
"""

L_LOG_MAX = 32
LOG_FEAT_DIM = 6  # log_type, player_rel, card_id, area_from, area_to, scalar

#: Highest ``cg.api.AreaType`` value (LOOKING=12).  Areas are encoded as
#: ``area + 1`` so that "no area" (-1) lands on 0, giving MAX_AREA + 2 rows.
MAX_AREA = 12
N_AREA_EMB = MAX_AREA + 2  # 0 = none, 1..13 = AreaType 0..12
