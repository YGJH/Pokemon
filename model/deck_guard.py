"""Inference-time anti-deck-out guard (agent side only — no retraining).

Archetype 16 (Dragapult ex) wins 63.3% of its corpus games, yet 10.3% of its
losses are deck-outs: with 1-3 cards left it still plays Poké Pad /
Buddy-Buddy Poffin and loses to the next start-of-turn draw (engine RESULT
reason 2, ``cg/api.py``).  Two inference-time mechanisms attack this without
touching the featurizer, the shards, or any checkpoint:

- **Mechanism A (hard mask).**  An option whose resolution leaves the deck
  empty before the next-turn draw is mask-illegal, exactly like the engine's
  own legality mask — it can never be un-learned by the network, and it costs
  nothing at inference.  Prize-winning attacks are exempt (a KO on the last
  prize ends the game before the deck matters), and the mask never empties
  the legal set (the least-bad option survives).
- **Mechanism B (tie-break).**  Only when the deck is low *and* the hand is
  large, options within ``margin`` logits of the argmax are re-ranked by net
  deck delta — the "same board progress, keep more deck" preference.  A KO
  attack on top is never overridden: taking prizes always outranks deck
  conservation.  ``GuardStats`` records how often the tie-break actually
  changes the pick, so its value is measured rather than assumed.

Why an explicit card table: ``card_draw_counts`` (static cols 83:85) only
parses "draw" phrasing, so search cards (Poké Pad, Ultra Ball, Poffin,
Crispin, Dawn) read 0, Judge is deliberately skipped by the regexes, and
Lillie's Determination parses as draw-6 while the shuffle-your-hand-back
clause is invisible — the wrong sign exactly in the big-hand states where
mechanism B fires.  The table below is transcribed from EN_Card_Data.csv;
cols 83:85 remain the fallback for unlisted cards (burn-only, conservative).

Deltas are in cards, relative to *my* deck: negative burns, positive refills.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Static card-row layout (ptcg_mine/cards.py, ptcg_il/featurizer.py).  Repeated
# here rather than imported so this module stays self-contained and is vendored
# into the submission bundle verbatim (like ref_map.py).
CARD_DRAW_FIXED_COL = 83
CARD_DRAW_TO_HAND_COL = 84
DRAW_N = 10

# Engine OptionType values (cg/api.py).
_OT_NUMBER, _OT_YES, _OT_NO, _OT_CARD = 0, 1, 2, 3
_OT_PLAY, _OT_ABILITY, _OT_ATTACK, _OT_SKILL = 7, 10, 13, 15

# SelectType.MAIN — the only context the guard acts in.  MAIN never carries
# CARD options (cg/api.py:56), so every PLAY/ABILITY/SKILL option there is
# mine by construction: PLAY is a bare index into *my* hand, and the engine
# does not offer the opponent's abilities.  Search-pick selects (CARD,
# ATTACHED_CARD, …) are left alone because their deck cost was already
# charged when the search card itself was played.
_SEL_MAIN = 0

#: card_id -> (kind, n, n_at_6_prizes), transcribed from EN_Card_Data.csv:
#:   "burn"         delta = -n                          (search n / draw n)
#:   "shuffle_draw" delta = +hand - n (n_at_6_prizes when prizes == 6)
#: Poké Pad 1152 "Search your deck for a Pokémon"; Ultra Ball 1121 idem;
#: Buddy-Buddy Poffin 1086 "up to 2 Basic Pokémon"; Crispin 1198 two energy;
#: Dawn 1231 "a Basic, a Stage 1 and a Stage 2"; Lillie's Determination 1227
#: "Shuffle your hand into your deck. Then, draw 6 (8 at exactly 6 prizes)";
#: Judge 1213 "each player shuffles their hand and draws 4";
#: Unfair Stamp 1080 "each player shuffles their hand; you draw 5".
#:
#: The split by where the effect resolves is load-bearing: a Pokémon's burn
#: belongs to its *ability* (SKILL/ABILITY option) — playing that Pokémon
#: from hand to the bench touches no deck cards, and charging it there masks
#: real plays at low deck (caught in the episode replay).
TRAINER_BURN_TABLE: dict[int, tuple[str, int, int]] = {
    1227: ("shuffle_draw", 6, 8),   # Lillie's Determination
    1213: ("shuffle_draw", 4, 4),   # Judge
    1080: ("shuffle_draw", 5, 5),   # Unfair Stamp
    1086: ("burn", 2, 2),           # Buddy-Buddy Poffin
    1198: ("burn", 2, 2),           # Crispin
    1231: ("burn", 3, 3),           # Dawn
    1152: ("burn", 1, 1),           # Poké Pad
    1121: ("burn", 1, 1),           # Ultra Ball
}

#: Pokémon whose *ability* draws/searches: Fezandipiti ex 140 Flip the Script
#: "draw 3"; Drakloak 120 Recon Directive "top 2, put 1 into your hand".
ABILITY_BURN_TABLE: dict[int, tuple[str, int, int]] = {
    140: ("burn", 3, 3),            # Fezandipiti ex (Flip the Script)
    120: ("burn", 1, 1),            # Drakloak (Recon Directive)
}

#: Union view (id-existence checks, docs).  Scoped dispatch uses the splits.
BURN_TABLE: dict[int, tuple[str, int, int]] = {
    **TRAINER_BURN_TABLE, **ABILITY_BURN_TABLE,
}

# Card-type one-hot lives at static row [2:9] (ptcg_mine/cards.py:107);
# engine CardType: POKEMON=0, ITEM=1, SUPPORTER=3 (cg/api.py:39-46).
_CARD_TYPE_COL = 2
_CARD_TYPE_ITEM = 1
_CARD_TYPE_SUPPORTER = 3


@dataclass(frozen=True)
class GuardConfig:
    """Thresholds, both corpus-measured (delivery report):

    ``deck_low`` = 10 is the p90 of the per-turn net deck-consumption
    distribution measured over the full 93,609-game corpus (1.75M turn
    transitions: median 2, mean 4.0, p90 = 10, p95 = 14) — below it one
    typical turn can empty the deck.  ``margin`` = 0.25 is ~p25 of the
    masked top-2 logit-gap distribution over 2,226 gated arch-16 val
    decisions (p25 = 0.133, p50 = 0.496), re-ranking only genuine near-ties
    (13.3% of gated decisions, all inspected examples sensible).
    """

    deck_low: int = 10
    hand_gate: int = 6
    margin: float = 0.25
    enable_a: bool = True
    enable_b: bool = True


@dataclass
class GuardStats:
    """How the guard actually fired — the evidence for mechanism B's value."""

    a_masked: int = 0      # options mechanism A masked, total
    a_decisions: int = 0   # MAIN decisions mechanism A inspected
    b_gated: int = 0       # decisions where B's gate opened
    b_changed: int = 0     # decisions where B changed the final pick
    recent: deque = field(default_factory=lambda: deque(maxlen=50))


def card_deck_delta(card_id, *, hand: int, prizes: int, card_row=None,
                    otype: int | None = None) -> int:
    """Net deck delta of resolving card ``card_id`` (negative burns my deck).

    The explicit table wins; unlisted cards fall back to the static draw
    columns, interpreted burn-only (a shuffle-back is never assumed for a
    card we do not know).  Unresolvable cards score 0 rather than an invented
    cost — the same rule the featurizer's dim-13 block follows.

    ``otype`` scopes the lookup to where the effect resolves: trainer burns
    fire only on PLAY (a trainer's text *is* its play effect), ability burns
    only on ABILITY/SKILL — and the PLAY fallback only reads trainer rows,
    because a Pokémon's draw columns describe its ability, not its play.
    ``otype=None`` (table-only callers) checks both scopes.
    """
    cid = int(card_id)
    entry = None
    if otype is None or otype == _OT_PLAY:
        entry = TRAINER_BURN_TABLE.get(cid)
    if entry is None and (otype is None or otype in (_OT_ABILITY, _OT_SKILL)):
        entry = ABILITY_BURN_TABLE.get(cid)
    if entry is not None:
        kind, n, n6 = entry
        if kind == "shuffle_draw":
            return int(hand) - (n6 if int(prizes) == 6 else n)
        return -n
    if card_row is None:
        return 0
    if otype == _OT_PLAY and not (
            card_row[_CARD_TYPE_COL + _CARD_TYPE_ITEM] > 0
            or card_row[_CARD_TYPE_COL + _CARD_TYPE_SUPPORTER] > 0):
        return 0
    d_fixed = float(card_row[CARD_DRAW_FIXED_COL]) * DRAW_N
    d_to_hand = float(card_row[CARD_DRAW_TO_HAND_COL]) * DRAW_N
    burn = d_fixed + max(0.0, d_to_hand - int(hand))
    return -int(round(burn))


def option_deck_deltas(*, opt_type, opt_card_id, opt_atk_risk, deck: int,
                       hand: int, prizes: int,
                       engine_card_features=None) -> np.ndarray:
    """int64[O] net deck delta per option; everything outside the guard's
    scope (END, CARD picks, NUMBER, …) scores 0.

    ATTACK options come from ``opt_scalar[:, 8]`` (the featurizer's
    draw-estimate/deck ratio) rather than the card row, whose draw text
    describes the Pokémon's *ability*, not the attack — the same split the
    featurizer makes between dims 8 and 13.
    """
    n = len(opt_type)
    out = np.zeros(n, dtype=np.int64)
    for j in range(n):
        ot = int(opt_type[j])
        if ot in (_OT_PLAY, _OT_ABILITY, _OT_SKILL):
            cid = int(opt_card_id[j])
            row = (engine_card_features.get(cid)
                   if engine_card_features is not None else None)
            out[j] = card_deck_delta(cid, hand=hand, prizes=prizes,
                                     card_row=row, otype=ot)
        elif ot == _OT_ATTACK:
            risk = float(opt_atk_risk[j])
            out[j] = -int(round(risk * deck)) if risk < 1.0 else -int(deck)
    return out


def deckout_kill_mask(*, opt_type, opt_card_id, opt_ko, opt_atk_risk,
                      opt_mask, deck: int, hand: int, prizes: int,
                      engine_card_features=None) -> np.ndarray:
    """bool[O], True where the option is legal but provably decks me out.

    ``post <= 0`` means the deck is empty before the next start-of-turn draw
    — a certain loss unless the game ends first.  Only genuine burns
    (``delta < 0``) kill: at deck 0 a delta-0 option is not what ends you,
    and masking everything would just turn argmax into noise.  The exemption
    is an ATTACK that KOs for the last prize (``opt_scalar[:, 7] == 1.0``
    with ``prizes == 1``).  The kill set never covers every legal option:
    losing next turn is only *nearly* certain, while an empty mask turns
    argmax into an arbitrary engine-legal pick, so the highest-``post``
    option survives.
    """
    deltas = option_deck_deltas(
        opt_type=opt_type, opt_card_id=opt_card_id, opt_atk_risk=opt_atk_risk,
        deck=deck, hand=hand, prizes=prizes,
        engine_card_features=engine_card_features)
    post = int(deck) + deltas
    kill = np.asarray(opt_mask, dtype=bool) & (deltas < 0) & (post <= 0)
    exempt = ((opt_type == _OT_ATTACK) & (int(prizes) == 1)
              & (opt_ko >= 1.0 - 1e-6))
    kill &= ~exempt
    if kill.any() and not (np.asarray(opt_mask, dtype=bool) & ~kill).any():
        legal_idx = np.flatnonzero(opt_mask)
        kill[legal_idx[np.argmax(post[legal_idx])]] = False
    return kill


def tiebreak_pick(*, logits, opt_mask, opt_type, opt_card_id, opt_ko,
                  opt_atk_risk, deck: int, hand: int, prizes: int,
                  engine_card_features, config: GuardConfig) -> int:
    """Argmax, except that within ``config.margin`` of the top logit the
    highest-deck-delta candidate wins.  Gate and priority rules, in order:

    1. Gate closed (deck above ``deck_low`` or hand at/below ``hand_gate``)
       → plain argmax; the deck is not under pressure.
    2. Top is a KO attack → keep it; taking prizes always outranks deck
       conservation (mechanism B may never override a prize push).
    3. Otherwise the highest-delta candidate within the margin wins, with
       ties (including the top's own delta) keeping the top.
    """
    masked = np.where(opt_mask, logits, -np.inf)
    top = int(np.argmax(masked))
    if not (deck <= config.deck_low and hand > config.hand_gate):
        return top
    if int(opt_type[top]) == _OT_ATTACK and float(opt_ko[top]) >= 1.0 - 1e-6:
        return top
    deltas = option_deck_deltas(
        opt_type=opt_type, opt_card_id=opt_card_id, opt_atk_risk=opt_atk_risk,
        deck=deck, hand=hand, prizes=prizes,
        engine_card_features=engine_card_features)
    cand = np.flatnonzero(opt_mask & (masked >= masked[top] - config.margin))
    best = deltas[cand].max()
    winners = cand[deltas[cand] == best]
    return top if top in winners else int(winners[0])


class DeckGuard:
    """Batch-dict adapter used by the submission template and PolicyAgent.

    Holds the static table, config, and cumulative stats.  ``apply_mask`` is
    mechanism A (in-place ``opt_mask`` edit, so both the single-select
    ``masked_fill`` and ``select_multi``'s cloned picked-mask honor it);
    ``pick`` is mechanism B for single-select decisions.
    """

    def __init__(self, config: GuardConfig | None = None,
                 engine_card_features=None,
                 stats: GuardStats | None = None) -> None:
        self.config = config or GuardConfig()
        self.engine_card_features = engine_card_features
        self.stats = stats or GuardStats()

    def gate_open(self, *, sel_type: int, max_count: int, deck: int,
                  hand: int) -> bool:
        return (int(sel_type) == _SEL_MAIN and int(max_count) == 1
                and deck <= self.config.deck_low
                and hand > self.config.hand_gate)

    @staticmethod
    def _np(batch: dict, key: str) -> np.ndarray:
        return batch[key][0].detach().cpu().numpy()

    def _arrays(self, batch: dict):
        opt_scalar = self._np(batch, "opt_scalar")
        return dict(
            opt_type=self._np(batch, "opt_type"),
            opt_card_id=self._np(batch, "opt_card_id"),
            opt_ko=opt_scalar[:, 7],
            opt_atk_risk=opt_scalar[:, 8],
        )

    def apply_mask(self, batch: dict, *, sel_type: int, deck: int, hand: int,
                   prizes: int) -> None:
        """Mechanism A: edit ``batch["opt_mask"]`` in place (MAIN selects).

        ``sel_type`` is taken explicitly rather than from the batch because
        live_eval's ``_sample_to_batch`` drops 0-d scalars — a silent missing
        key here would disable the guard without a trace.
        """
        if not self.config.enable_a:
            return
        if int(sel_type) != _SEL_MAIN:
            return
        self.stats.a_decisions += 1
        kill = deckout_kill_mask(
            **self._arrays(batch),
            opt_mask=self._np(batch, "opt_mask"),
            deck=deck, hand=hand, prizes=prizes,
            engine_card_features=self.engine_card_features)
        if kill.any():
            import torch

            mask = batch["opt_mask"][0]
            mask &= ~torch.from_numpy(kill).to(mask.device)
            self.stats.a_masked += int(kill.sum())

    def pick(self, logits, batch: dict, *, sel_type: int, max_count: int,
             deck: int, hand: int, prizes: int) -> int:
        """Mechanism B pick for single-select; plain argmax when ungated."""
        opt_mask = self._np(batch, "opt_mask")
        masked = np.where(opt_mask, self._np_from_logits(logits), -np.inf)
        top = int(np.argmax(masked))
        if not self.config.enable_b:
            return top
        if not self.gate_open(sel_type=sel_type, max_count=max_count,
                              deck=deck, hand=hand):
            return top
        self.stats.b_gated += 1
        p = tiebreak_pick(
            logits=masked, opt_mask=opt_mask, **self._arrays(batch),
            deck=deck, hand=hand, prizes=prizes,
            engine_card_features=self.engine_card_features,
            config=self.config)
        if p != top:
            self.stats.b_changed += 1
            self.stats.recent.append({
                "deck": int(deck), "hand": int(hand), "old": top, "new": p,
            })
        return p

    @staticmethod
    def _np_from_logits(logits) -> np.ndarray:
        return logits[0].detach().cpu().numpy()
