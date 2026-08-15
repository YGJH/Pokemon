"""Effect keywords mined from card and attack oracle text.

``card_static_row`` is otherwise entirely numeric, so two cards with the same HP,
type and attack damage are *bit-identical* to the model — and the policy has no
learned card-id embedding to fall back on.  Measured on the real engine tables,
all 61 Supporters share one feature row: ``Boss's Orders`` is indistinguishable
from ``Cheren``.  This module turns the text those cards do carry into a fixed
binary multihot so the encoder can tell them apart.

**The tuple index is the feature column.**  ``KEYWORDS`` is therefore frozen and
append-only: inserting a keyword in the middle silently repoints every later
column of every trained checkpoint, exactly like renumbering archetype cluster
ids.  Retire a keyword by leaving its slot in place as a dead always-zero column.

Patterns are matched case-insensitively.  This is load-bearing, not cosmetic:
oracle text capitalises the leading verb, so a case-sensitive ``draw \\d+ card``
misses ``"Draw 3 cards."``  A prototype without ``re.IGNORECASE`` left 33 of the
61 Supporters with an all-zero row.
"""

import re
from typing import Iterable

import numpy as np

#: ``(name, pattern)`` pairs.  Index == feature column.  APPEND ONLY.
KEYWORDS: tuple[tuple[str, str], ...] = (
    ("draw",             r"draw \d+ (?:more )?card|draw a card|draw cards|draw that many|draw up to|draw cards until"),
    ("search_deck",      r"search your deck"),
    ("deck_look",        r"look at the top \d+ card|look at the top card"),
    ("discard_own",      r"discard your hand|discard (?:a|an|another|\d+|all|up to|the top|that|this|those|your|the other)\b|discard it\b|discard them\b"),
    ("discard_opp",      r"your opponent discards|opponent['’]?s? hand.{0,60}discard|discard .{0,60}from (?:\d+ of )?your opponent"),
    ("hand_disrupt",     r"shuffles? their hand into their deck|shuffle your hand into your deck|opponent['’]?s? hand.{0,40}(?:shuffle|reveal)"),
    ("gust",             r"switch in \d+ of your opponent|switch out your opponent|opponent.{0,40}benched pok.mon to the active"),
    ("switch_own",       r"switch (?:this|your) (?:pok.mon|active)|switch \d+ of your"),
    ("heal",             r"\bheal\b|remove.{0,25}damage counter"),
    ("place_damage",     r"put \d+ damage counter|put damage counter|place \d+ damage counter"),
    ("bench_damage",     r"benched pok.mon"),
    ("bench_accel",      r"onto your bench|put .{0,40}onto (?:your|that) bench"),
    ("evolve_effect",    r"to evolve it|evolve.{0,30}during your first turn|devolve"),
    ("status_offensive", r"\bpoisoned\b|\bburned\b"),
    ("status_lock",      r"\basleep\b|\bparalyzed\b|\bconfused\b"),
    ("energy_accel",     r"attach (?:a|an|\d+|up to|basic|the other|that|it|them)\b.{0,90}(?:energy|pok.mon)|attach .{0,30}energy card.{0,60}(?:to|onto)"),
    ("energy_deny",      r"discard.{0,60}energy from (?:\d+ of )?your opponent|opponent.{0,80}discard.{0,30}energy|move an energy|discard .{0,30}special energy"),
    ("prevent_damage",   r"prevent all damage|takes? no damage|prevent all effects|damage done to this pok.mon.{0,40}reduced|-\d+ damage from attacks|do(?:es)? \d+ less damage"),
    ("damage_scaling",   r"do(?:es)? \d+ more damage|damage for each|does \d+ damage times|\d+ more damage"),
    ("coin_flip",        r"flip \d+ coins|flip a coin"),
    ("ability_lock",     r"have no abilities|can['’]?t use.{0,25}abilit"),
    ("prize_effect",     r"prize card"),
    ("retreat_effect",   r"retreat cost|retreating|retreat for"),
    ("once_per_turn",    r"once during your turn"),
    ("target_pokemon",   r"(?:search|look at|reveal|put|shuffle)[^.]{0,80}\bpok.mon\b"),
    ("target_energy",    r"(?:search|look at|reveal|put|attach)[^.]{0,80}energy card"),
    ("target_trainer",   r"\btrainer card|\bsupporter card|\bitem card|\bstadium card|pok.mon tool"),
    ("to_deck",          r"into your deck|on top of (?:it|your deck)|bottom of your deck|back into your deck"),
    ("recover_discard",  r"from your discard pile"),
)

KEYWORD_NAMES: tuple[str, ...] = tuple(name for name, _ in KEYWORDS)
K_EFFECT: int = len(KEYWORDS)

_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for _, pattern in KEYWORDS
)

assert len(KEYWORD_NAMES) == len(set(KEYWORD_NAMES)), "duplicate keyword name"


def effect_keyword_row(texts: Iterable[str]) -> np.ndarray:
    """float32[K_EFFECT] binary multihot over the concatenation of *texts*.

    Binary rather than a count: a card that draws twice is not twice the draw
    card, and a count would need a normalizer no other column in the row uses.
    """
    blob = "\n".join(t for t in texts if t)
    row = np.zeros(K_EFFECT, dtype=np.float32)
    if not blob:
        return row
    for i, pattern in enumerate(_PATTERNS):
        if pattern.search(blob):
            row[i] = 1.0
    return row


def ability_keyword_row(card) -> np.ndarray:
    """float32[K_EFFECT] over every ``card.skills[].text``."""
    return effect_keyword_row(s.text for s in (getattr(card, "skills", None) or []))


def attack_keyword_row(attack) -> np.ndarray:
    """float32[K_EFFECT] over ``attack.text``."""
    return effect_keyword_row([getattr(attack, "text", "") or ""])


# ============================================================
# Draw-count parsing (for attack_static_row numerics)
# ============================================================

_DRAW_FIXED_PAT = re.compile(r"draw (\d+) (?:more )?card", re.IGNORECASE)
_DRAW_TO_HAND_PAT = re.compile(
    r"(?:draw cards until you have|you may draw cards until you have) (\d+) card",
    re.IGNORECASE,
)
_DRAW_A_CARD_PAT = re.compile(r"draw a card", re.IGNORECASE)
_DRAW_BOTH_PAT = re.compile(r"each player draws (\d+) card", re.IGNORECASE)


def draw_fixed(attack) -> int:
    """Explicit draw count from attack oracle text, or 0.

    Matches ``draw 2 cards``, ``Draw a card.`` (implicit 1), and
    ``each player draws N``.  Does *not* match draw-to-hand-size forms.
    """
    text = getattr(attack, "text", "") or ""
    m = _DRAW_FIXED_PAT.search(text)
    if m:
        return int(m.group(1))
    if _DRAW_A_CARD_PAT.search(text):
        return 1
    m = _DRAW_BOTH_PAT.search(text)
    if m:
        return int(m.group(1))
    return 0


def draw_to_hand(attack) -> int:
    """Target hand size in 'draw cards until you have N cards', or 0."""
    text = getattr(attack, "text", "") or ""
    m = _DRAW_TO_HAND_PAT.search(text)
    if m:
        return int(m.group(1))
    return 0


# ============================================================
# Bench-damage parsing (for attack_static_row[16])
# ============================================================

# An attack's ``damage`` field is **always** the number dealt to the Active.
# Measured over the engine's attack pool, 27 attacks also reach a benched
# Pokémon and every one of them states that figure only in the oracle text, in
# one of three forms.  Using the ``damage`` field against a bench slot therefore
# over-reports for all of them -- Phantom Dive reads 200 against a 70 HP bench
# Pokémon (ratio 2.0, KO flag set) when the truth is at most 60.
#
# Note the `also` forms and the bare form are both needed: Pinpoint Dive is a
# *pure* snipe whose ``damage`` field is 0 and whose whole output is on the
# bench.
_BENCH_DMG_PATS = (
    # "This attack (also) does 30 damage to 1 of your opponent's Benched Pokemon"
    re.compile(r"does (\d+) damage to (?:1 of )?(?:your opponent[’']?s )?Benched", re.I),
    # "... does 50 damage to each (of your opponent's) Benched Pokemon"
    re.compile(r"does (\d+) damage to each (?:of your opponent[’']?s )?Benched", re.I),
    re.compile(r"does (\d+) damage to each Benched", re.I),
)

#: A damage counter is 10 HP.  "Put N damage counters on your opponent's
#: Benched Pokemon in any way you like" means up to N*10 on a *single* target,
#: which is the number a KO check needs.
_BENCH_COUNTER_PAT = re.compile(
    r"put (\d+) damage counters? on (?:your opponent[’']?s )?Benched", re.I
)
_BENCH_COUNTER_WORD_PAT = re.compile(
    r"put (a|one|two|three|four|five|six) damage counters? on "
    r"(?:your opponent[’']?s )?Benched", re.I
)
_WORD_TO_INT = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def attack_bench_damage(attack) -> int:
    """Damage this attack can put on a **single benched** Pokémon, in HP.

    Zero when the attack does not reach the bench.  Deliberately the *maximum*
    a single bench target can take: for "put N damage counters ... in any way
    you like" that is all N counters aimed at one Pokémon, which is exactly the
    quantity a "can I KO it" check needs.

    Text that merely *mentions* the bench -- moving Energy to a benched Pokémon,
    searching a Pokémon onto the bench -- must not match, so the patterns all
    require an explicit damage or damage-counter verb.
    """
    text = getattr(attack, "text", "") or ""
    if not text:
        return 0
    for pat in _BENCH_DMG_PATS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    m = _BENCH_COUNTER_PAT.search(text)
    if m:
        return int(m.group(1)) * 10
    m = _BENCH_COUNTER_WORD_PAT.search(text)
    if m:
        return _WORD_TO_INT[m.group(1).lower()] * 10
    return 0


class _TextCarrier:
    """Adapts a blob of text to the ``.text`` interface the parsers expect."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def card_draw_counts(card) -> tuple[int, int]:
    """``(draw_fixed, draw_to_hand)`` over every ``card.skills[].text``.

    Trainers keep their oracle text in ``skills[].text`` exactly as Pokémon
    abilities do -- Boss's Orders and Colress's Tenacity are ``cardType`` 3 with
    one skill each -- so the two attack parsers apply unchanged.  Measured over
    the engine's 1267 cards: 42 mention "draw" and **38 (90%) yield a numeric
    count**; the four that do not (Judge, Frosmoth, Meddling Memo, Mystery
    Garden) phrase it in forms the patterns deliberately skip and fall back to
    the binary ``draw`` keyword already in the ability row.

    This is what lets a *per-option* deck cost be a number rather than a flag:
    without it the featurizer can only say "this card draws something".
    """
    skills = getattr(card, "skills", None) or []
    blob = _TextCarrier("\n".join((getattr(s, "text", "") or "") for s in skills))
    return draw_fixed(blob), draw_to_hand(blob)
