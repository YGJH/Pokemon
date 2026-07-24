"""Phase 4 QA gates — validation checks before training (D.5).

Provides independent check functions for each gate plus a top-level
``run_qa_checks`` orchestrator that runs them all and returns a results dict.
Critical failures (coverage, label sanity) raise ``AssertionError``;
non-critical warnings are logged.

Usage::

    results = run_qa_checks(
        shard_dir="data/shards",
        meta_path="data/meta.parquet",
        vocab=vocab_dict,
        fixed_deck=fixed_deck,
    )
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Deck legality: CardType enum values (verified against engine `all_card_data()`)
CARDTYPE_BASIC_ENERGY = 5


# ============================================================
# Public entry point
# ============================================================


def run_qa_checks(
    shard_dir: str | Path,
    meta_path: str | Path,
    vocab: dict | None = None,
    vocab_path: str | Path | None = None,
    fixed_deck: list[int] | None = None,
    all_episode_deck_cards: list[int] | None = None,
    *,
    n_vocab_values: list[int] | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run all QA gates and return a flat results dict.

    Parameters
    ----------
    shard_dir : Path
        Directory containing ``*.npz`` shard files.
    meta_path : Path
        Path to ``meta.parquet``.
    vocab : dict or None
        Vocab dict (as produced by ``ptcg_mine.vocab.build_vocab``).  If None,
        *vocab_path* must be given.
    vocab_path : Path or None
        Alternative — load vocab from this JSON file.
    fixed_deck : list[int] or None
        The FIXED_DECK (60 card ids).  If None, deck legality check is skipped.
    all_episode_deck_cards : list[int] or None
        All raw card ids appearing in any sampled episode's decks.  If None and
        fixed_deck is provided, coverage is only checked for fixed_deck cards.
    n_vocab_values : list[int] or None
        List of N_VOCAB cutoffs to evaluate in the OOV coverage curve.
        Default: [200, 300, 400, 500].
    max_samples : int or None
        Cap the number of shard samples read (for speed on large corpora).

    Returns
    -------
    dict
        Flat results dict with keys like ``coverage_pass``,
        ``n_variable_length``, ``n_collision_options``, ``label_sanity_pass``,
        ``won_count``, ``lost_count``, ``deck_legality_pass``, plus detailed
        counts and per-gate results.
    """
    shard_dir = Path(shard_dir)
    meta_path = Path(meta_path)
    results: dict[str, Any] = {}

    # Load vocab
    if vocab is None:
        if vocab_path is None:
            raise ValueError("Either vocab or vocab_path must be provided")
        vocab = _load_vocab(vocab_path)
    id_to_index = vocab.get("id_to_index", {})
    # Convert string keys to int if needed
    id_to_index = {int(k): int(v) for k, v in id_to_index.items()}

    # Load meta
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
    else:
        logger.warning("meta.parquet not found; some QA gates will be skipped.")
        meta = pd.DataFrame()

    # --- Coverage (D.5.1) ---
    if all_episode_deck_cards is not None:
        cov_pass, cov_missing, cov_total = check_coverage(all_episode_deck_cards, id_to_index)
        results["coverage_pass"] = cov_pass
        results["coverage_missing"] = cov_missing
        results["coverage_total_kept_cards"] = cov_total
    elif fixed_deck is not None:
        cov_pass, cov_missing, cov_total = check_coverage(fixed_deck, id_to_index)
        results["coverage_pass"] = cov_pass
        results["coverage_missing"] = cov_missing
        results["coverage_total_kept_cards"] = cov_total
        results["coverage_note"] = "Checked only FIXED_DECK cards (all_episode_deck_cards not provided)"
    else:
        results["coverage_pass"] = None
        results["coverage_skipped"] = True

    # --- OOV coverage curve (D.5.2) ---
    if all_episode_deck_cards is not None:
        nv_vals = n_vocab_values or [200, 300, 400, 500]
        oov_curve = check_oov_coverage_curve(all_episode_deck_cards, nv_vals)
        results["oov_coverage_curve"] = oov_curve
    else:
        results["oov_coverage_curve"] = None

    # --- Variable-length multi-select (D.5.3) ---
    if not meta.empty:
        vl_report = check_variable_length_multi_select(meta)
        results.update(vl_report)
    else:
        results["n_variable_length"] = 0
        results["variable_length_share"] = 0.0

    # --- Attachment-collision audit (D.5.4) ---
    if shard_dir.exists() and not meta.empty:
        ac_report = check_attachment_collision(shard_dir, meta, max_samples=max_samples)
        results.update(ac_report)
    else:
        results["n_collision_options"] = 0
        results["collision_share"] = 0.0

    # --- Label sanity (D.5.5) ---
    if shard_dir.exists():
        ls_pass, ls_details = check_label_sanity(shard_dir, meta, max_samples=max_samples)
        results["label_sanity_pass"] = ls_pass
        results["label_sanity_details"] = ls_details
    else:
        results["label_sanity_pass"] = None

    # --- Outcome balance (D.5.6) ---
    if not meta.empty:
        ob_report = check_outcome_balance(meta)
        results.update(ob_report)
    else:
        results["won_count"] = 0
        results["lost_count"] = 0

    # --- Reference round-trip (D.5.x) ---
    obs_path: Path | None = None
    if shard_dir is not None:
        candidate = Path(shard_dir).parent / "observations"
        if candidate.is_dir():
            obs_path = candidate
    rt_pass, rt_details = check_reference_roundtrip(shard_dir, obs_path, max_samples=max_samples)
    results["reference_roundtrip_pass"] = rt_pass
    results["reference_roundtrip_details"] = rt_details

    # --- Deck legality (D.5.7) ---
    if fixed_deck is not None:
        dl_pass, dl_details = check_deck_legality(fixed_deck)
        results["deck_legality_pass"] = dl_pass
        results["deck_legality_details"] = dl_details
    else:
        results["deck_legality_pass"] = None

    # --- Balance report (D.5.8) ---
    if not meta.empty:
        br = balance_report(meta)
        results.update(br)

    return results


# ============================================================
# Individual QA gates
# ============================================================


def check_reference_roundtrip(
    shard_dir: str | Path | None = None,
    obs_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> tuple[bool | None, dict[str, Any]]:
    """Verify opt_src_idx / opt_card_id resolve to the card the option text implies.

    For a random sample of options, we decode opt_src_idx against the ref_map
    derived from the observation and confirm that the card at that index has
    the same cardId as opt_card_id.  This guards against off-by-one errors in
    the featurizer's pointer encoding.

    Parameters
    ----------
    shard_dir : Path or None
        Directory with shard npz files (to read opt_src_idx / opt_card_id).
    obs_dir : Path or None
        Directory with raw observation dicts corresponding to shard samples.
        If None, the gate cannot run and returns ``(None, skipped_details)``.
    max_samples : int or None
        Max number of samples to check.

    Returns
    -------
    (passed, details)
        passed is True/False if the gate ran, None if skipped.
    """
    details: dict[str, Any] = {"checked": 0, "errors": 0, "skipped": False}
    if shard_dir is None or obs_dir is None:
        details["skipped"] = True
        details["note"] = (
            "Reference round-trip check skipped: raw observation data not available. "
            "This gate requires the original game-state observations to verify that "
            "opt_src_idx/opt_card_id resolve correctly.  Place observation dicts in "
            "data/observations/ to enable this gate."
        )
        logger.info("Reference round-trip QA gate: %s", details["note"])
        return None, details

    shard_dir = Path(shard_dir)
    obs_dir = Path(obs_dir)
    import json

    shard_files = sorted(shard_dir.glob("*.npz"))
    samples_checked = 0
    errors = 0
    missing_obs = 0

    for sf in shard_files:
        try:
            data = np.load(sf, mmap_mode="r", allow_pickle=False)
        except OSError:
            continue
        if "opt_src_idx" not in data or "opt_card_id" not in data or "opt_mask" not in data:
            continue

        opt_src = data["opt_src_idx"]     # [S, O_MAX]
        opt_card = data["opt_card_id"]     # [S, O_MAX]
        opt_mask = data["opt_mask"]        # [S, O_MAX]

        # Look for corresponding observation file
        obs_file = obs_dir / f"{sf.stem}_obs.jsonl"
        if not obs_file.exists():
            missing_obs += 1
            continue

        # Read observations for this shard
        observations: list[dict] = []
        try:
            with open(obs_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        observations.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            missing_obs += 1
            continue

        for s in range(opt_src.shape[0]):
            if s >= len(observations):
                break
            obs_dict = observations[s]
            n_opts = int(opt_mask[s].sum())
            if n_opts == 0:
                continue

            # Build a simple ref_map from the observation to resolve src_idx → cardId
            ref_cards = _build_simple_ref_from_obs(obs_dict)
            if ref_cards is None:
                continue

            for o in range(n_opts):
                src_idx = int(opt_src[s, o])
                card_id = int(opt_card[s, o])

                # -1 src_idx means no source (e.g., text-only option)
                if src_idx < 0:
                    continue

                if src_idx in ref_cards:
                    expected_cid = ref_cards[src_idx]
                    if expected_cid != card_id:
                        errors += 1
                # If src_idx not in ref_cards, it's a different entity type
                # (e.g., energy, tool) — skip without counting as error

            samples_checked += 1
            if max_samples is not None and samples_checked >= max_samples:
                break

        if max_samples is not None and samples_checked >= max_samples:
            break

    details["checked"] = samples_checked
    details["errors"] = errors
    details["missing_obs"] = missing_obs

    if missing_obs > 0:
        details["note"] = (
            f"{missing_obs} observation file(s) missing; round-trip check is partial."
        )

    if samples_checked == 0:
        details["skipped"] = True
        details["note"] = (
            "Reference round-trip check skipped: no matching observation files found "
            "for the shard data.  Place observation dicts in data/observations/ as "
            "shardname_obs.jsonl to enable this gate."
        )
        logger.info("Reference round-trip QA gate: %s", details["note"])
        return None, details

    ok = errors == 0
    if not ok:
        logger.error(
            "REFERENCE ROUND-TRIP FAIL: %d errors in %d samples — opt_src_idx/opt_card_id "
            "mismatch detected.  The featurizer's pointer encoding may be incorrect.",
            errors,
            samples_checked,
        )

    return ok, details


def _build_simple_ref_from_obs(obs_dict: dict) -> dict[int, int] | None:
    """Build a minimal ref_map from an observation dict: src_idx → cardId.

    This is a simplified version of the full featurizer's ref_map logic
    (see ref_map.py).  We index the player's hand, bench, active, etc.
    to resolve source indices.

    Returns None if the observation structure is unrecognized.
    """
    try:
        your_index = obs_dict["yourIndex"]
        players = obs_dict["players"]
        ref: dict[int, int] = {}
        idx = 0

        # Hand
        for card in players[your_index].get("hand", []) or []:
            if isinstance(card, dict) and "cardId" in card:
                ref[idx] = int(card["cardId"])
                idx += 1

        # Bench
        for card in players[your_index].get("bench", []) or []:
            if isinstance(card, dict) and "cardId" in card:
                ref[idx] = int(card["cardId"])
                idx += 1

        # Active
        active = players[your_index].get("active")
        if active is not None and isinstance(active, dict) and "cardId" in active:
            ref[idx] = int(active["cardId"])
            idx += 1

        # Discard
        for card in players[your_index].get("discard", []) or []:
            if isinstance(card, dict) and "cardId" in card:
                ref[idx] = int(card["cardId"])
                idx += 1

        return ref if ref else None
    except (KeyError, IndexError, TypeError):
        return None


def check_coverage(
    card_ids: list[int],
    id_to_index: dict[int, int],
) -> tuple[bool, list[int], int]:
    """Check that every card id in *card_ids* is covered by the vocab.

    Returns ``(pass, missing_ids, total_unique)``.
    Raises ``AssertionError`` if any card is missing.
    """
    unique = set(card_ids)
    missing = sorted(cid for cid in unique if cid not in id_to_index)
    ok = len(missing) == 0
    if not ok:
        logger.error(
            "COVERAGE FAIL: %d/%d card ids missing from vocab: %s",
            len(missing),
            len(unique),
            missing[:20],
        )
        raise AssertionError(
            f"Vocab coverage failure: {len(missing)} of {len(unique)} "
            f"kept-game card ids are not in vocab. First 20: {missing[:20]}"
        )
    return ok, missing, len(unique)


def check_oov_coverage_curve(
    all_card_ids: list[int],
    n_vocab_values: list[int] | None = None,
) -> dict[int, float]:
    """Report the fraction of card appearances outside a top-N vocab cut.

    Parameters
    ----------
    all_card_ids : list[int]
        All raw card ids appearing in decks across the sampled corpus
        (flattened, with multiplicities).
    n_vocab_values : list[int] or None
        Cutoffs to evaluate (e.g. [200, 300, 400, 500]).

    Returns
    -------
    dict
        ``{n: oov_fraction}`` — fraction of *occurrences* that fall outside
        the top-N most frequent card ids.
    """
    n_vocab_values = n_vocab_values or [200, 300, 400, 500]
    counts: dict[int, int] = defaultdict(int)
    for cid in all_card_ids:
        counts[cid] += 1

    total = sum(counts.values())
    if total == 0:
        return {n: 0.0 for n in n_vocab_values}

    # Rank by descending frequency
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    result: dict[int, float] = {}
    for n in sorted(n_vocab_values):
        in_vocab = sum(c for _, c in ranked[:n])
        oov_frac = 1.0 - (in_vocab / total)
        result[n] = round(oov_frac, 6)
    return result


def check_variable_length_multi_select(meta: pd.DataFrame) -> dict[str, Any]:
    """Count multi-select samples where ``minCount < maxCount``.

    Returns a dict with ``n_variable_length`` count and ``variable_length_share``
    as fraction of all multi-select samples.
    """
    multi = meta[meta["maxCount"] > 1]
    if len(multi) == 0:
        return {"n_variable_length": 0, "variable_length_share": 0.0}

    variable = multi[multi["minCount"] < multi["maxCount"]]
    n_var = int(len(variable))
    share = n_var / len(multi) if len(multi) > 0 else 0.0

    if share > 0.0:
        logger.warning(
            "VARIABLE-LENGTH MULTI-SELECT: %d samples (%.2f%% of multi-select) "
            "have minCount < maxCount.  v1 fixed-length loop (B.7) is NOT valid; "
            "a STOP head is required in v1.",
            n_var,
            share * 100,
        )
    return {"n_variable_length": n_var, "variable_length_share": round(share, 6)}


def check_attachment_collision(
    shard_dir: str | Path,
    meta: pd.DataFrame,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Among ``ENERGY/TOOL_CARD/CARD`` selects, count options that share both
    ``opt_src_idx`` AND ``opt_card_id`` (truly indistinguishable to the pointer).

    Returns ``n_collision_options``, ``collision_share``, ``n_options_total``.
    """
    shard_dir = Path(shard_dir)
    # We only need to check attachment-type options.
    # Walk shard files, reading opt_type, opt_src_idx, opt_card_id.
    n_collision = 0
    n_total_attachment_options = 0
    samples_checked = 0

    shard_files = sorted(shard_dir.glob("*.npz"))
    for sf in shard_files:
        try:
            data = np.load(sf, mmap_mode="r", allow_pickle=False)
        except OSError:
            continue
        if "opt_type" not in data:
            continue
        opt_type = data["opt_type"]       # [S, O_MAX]
        opt_src = data["opt_src_idx"]     # [S, O_MAX]
        opt_card = data["opt_card_id"]    # [S, O_MAX]
        opt_mask = data["opt_mask"]       # [S, O_MAX]

        # Attachment OptionTypes: CARD(3), TOOL_CARD(4), ENERGY_CARD(5), ENERGY(6)
        att_mask = np.isin(opt_type, [3, 4, 5, 6]) & opt_mask

        for s in range(opt_type.shape[0]):
            row_mask = att_mask[s]
            if not row_mask.any():
                continue
            # Build (src_idx, card_id) pairs for this sample's attachment options
            pairs = list(zip(
                opt_src[s][row_mask].tolist(),
                opt_card[s][row_mask].tolist(),
            ))
            # Count how many unique pairs vs total
            n_total_attachment_options += len(pairs)
            unique_pairs = set(pairs)
            n_collision += len(pairs) - len(unique_pairs)

            samples_checked += 1
            if max_samples is not None and samples_checked >= max_samples:
                break

        if max_samples is not None and samples_checked >= max_samples:
            break

    share = n_collision / max(n_total_attachment_options, 1)

    if share > 0.0:
        logger.warning(
            "ATTACHMENT COLLISION: %d/%d attachment options (%.2f%%) "
            "share both opt_src_idx AND opt_card_id — truly indistinguishable "
            "to the pointer head. Consider promoting attachments to tokens (A.7).",
            n_collision,
            n_total_attachment_options,
            share * 100,
        )

    return {
        "n_collision_options": n_collision,
        "collision_share": round(share, 6),
        "n_attachment_options": n_total_attachment_options,
        "attachment_samples_checked": samples_checked,
    }


def check_label_sanity(
    shard_dir: str | Path,
    meta: pd.DataFrame | None = None,
    max_samples: int | None = None,
) -> tuple[bool, dict[str, int]]:
    """Verify every sample's action labels are sane:

    - ``0 ≤ action_idx[k] < len(option)`` for each pick k.
    - Picks are distinct (within the sample's action_len).
    - ``len(picks) == action_len``, and ``minCount ≤ action_len ≤ maxCount``.

    Raises ``AssertionError`` on any failure.

    Returns ``(True, details)`` with per-error-type counts.
    """
    shard_dir = Path(shard_dir)
    details: dict[str, int] = defaultdict(int)
    samples_checked = 0

    shard_files = sorted(shard_dir.glob("*.npz"))
    for sf in shard_files:
        try:
            data = np.load(sf, mmap_mode="r", allow_pickle=False)
        except OSError:
            continue

        required = {"action_idx", "action_len", "opt_mask", "minCount", "maxCount"}
        if not required.issubset(data.keys()):
            continue

        action_idx = data["action_idx"]    # [S, O_MAX]
        action_len = data["action_len"]    # [S]
        opt_mask = data["opt_mask"]        # [S, O_MAX]
        min_count = data["minCount"]       # [S]
        max_count = data["maxCount"]       # [S]

        for s in range(action_idx.shape[0]):
            n_opts = int(opt_mask[s].sum())
            a_len = int(action_len[s])
            picks = action_idx[s, :a_len]

            # Bound
            too_small = bool((picks < 0).any())
            too_big = bool((picks >= n_opts).any()) if n_opts > 0 else False
            if too_small:
                details["out_of_bounds_low"] += 1
            if too_big:
                details["out_of_bounds_high"] += 1

            # Distinct
            if len(set(picks.tolist())) < len(picks):
                details["not_distinct"] += 1

            # Length
            mc = int(min_count[s])
            xc = int(max_count[s])
            if a_len < mc or a_len > xc:
                details["wrong_length"] += 1

            samples_checked += 1
            if max_samples is not None and samples_checked >= max_samples:
                break

        if max_samples is not None and samples_checked >= max_samples:
            break

    total_errors = sum(details.values())
    details["samples_checked"] = samples_checked
    ok = total_errors == 0

    if not ok:
        logger.error("LABEL SANITY FAIL: %d total errors — %s", total_errors, dict(details))
        raise AssertionError(
            f"Label sanity failure: {total_errors} errors across {samples_checked} "
            f"samples. Details: {dict(details)}"
        )

    return ok, dict(details)


def check_outcome_balance(meta: pd.DataFrame) -> dict[str, int]:
    """Report won vs lost decision counts.

    Returns ``won_count``, ``lost_count``.  Does NOT assert — just reports.
    """
    won = int(meta["won"].sum())
    total = len(meta)
    lost = total - won

    if total > 0:
        won_pct = won / total
        if won_pct < 0.30 or won_pct > 0.70:
            logger.warning(
                "OUTCOME IMBALANCE: won=%.1f%% (%d/%d). "
                "Balance may be poor; check expert filter or corpus sampling.",
                won_pct * 100,
                won,
                total,
            )

    return {"won_count": won, "lost_count": lost}


def check_deck_legality(deck: list[int]) -> tuple[bool, list[str]]:
    """Check that *deck* is a legal Pokemon TCG deck.

    Rules:
    1. Exactly 60 cards.
    2. At most 4 copies of any non-basic-energy card id.
    3. At most 1 ACE SPEC card.
    4. At least 1 basic Pokémon.

    Uses the engine's ``all_card_data()`` to identify basic-energy cards,
    ACE SPEC cards, and basic Pokémon.  If the engine is unavailable, falls
    back to a heuristic (cardType==5 for basic energy, aceSpec flag for
    ACE SPEC, basic flag for basic Pokémon).

    Returns ``(passed, failure_messages)``.
    """
    failures: list[str] = []

    # Length
    if len(deck) != 60:
        failures.append(f"Deck has {len(deck)} cards, expected 60")

    # Load engine data to identify card properties
    card_type_map, ace_spec_ids, basic_pokemon_ids = _load_card_properties()

    # Count copies per id
    from collections import Counter
    counts = Counter(deck)

    n_ace_spec = 0
    n_basic_pokemon = 0

    for cid, ct in counts.items():
        is_basic_energy = card_type_map.get(cid) == CARDTYPE_BASIC_ENERGY

        # ACE SPEC check
        if cid in ace_spec_ids:
            n_ace_spec += ct

        # Basic Pokemon check
        if cid in basic_pokemon_ids:
            n_basic_pokemon += ct

        # Copy limit (skip basic energies)
        if not is_basic_energy and ct > 4:
            failures.append(f"Card {cid}: {ct} copies (max 4)")

    if n_ace_spec > 1:
        failures.append(f"Deck has {n_ace_spec} ACE SPEC cards (max 1)")

    if n_basic_pokemon == 0:
        failures.append("Deck has 0 basic Pokémon (need ≥1)")

    ok = len(failures) == 0
    if not ok:
        logger.error("DECK LEGALITY FAIL: %s", "; ".join(failures))

    return ok, failures


def balance_report(meta: pd.DataFrame) -> dict[str, Any]:
    """Report samples-per-sel_ctx and samples-per-archetype_self (D.5.8).

    Returns ``balance_sel_ctx`` (dict ctx→count), ``balance_arch_self``
    (dict arch→count), and ``rare_contexts`` (list of contexts below a
    reasonable threshold).
    """
    result: dict[str, Any] = {}

    if "sel_ctx" in meta.columns:
        ctx_counts = meta["sel_ctx"].value_counts().to_dict()
        result["balance_sel_ctx"] = {int(k): int(v) for k, v in ctx_counts.items()}
        # Flag contexts with fewer than 100 samples as "rare"
        rare = {ctx: cnt for ctx, cnt in result["balance_sel_ctx"].items() if cnt < 100}
        result["rare_contexts"] = rare
        if rare:
            logger.warning(
                "RARE CONTEXTS (<100 samples): %s — the policy will underlearn these.",
                rare,
            )

    if "archetype_self" in meta.columns:
        arch_counts = meta["archetype_self"].value_counts().to_dict()
        result["balance_arch_self"] = {int(k): int(v) for k, v in arch_counts.items()}

    return result


# ============================================================
# Internal helpers
# ============================================================


def _load_vocab(path: str | Path) -> dict:
    """Load vocab from a JSON file."""
    import json
    with open(path) as f:
        return json.load(f)


def _load_card_properties() -> tuple[dict[int, int], set[int], set[int]]:
    """Load card type, ACE SPEC, and basic Pokémon identifiers from the engine.

    Returns ``(card_type_map, ace_spec_ids, basic_pokemon_ids)``.
    """
    card_type_map: dict[int, int] = {}
    ace_spec_ids: set[int] = set()
    basic_pokemon_ids: set[int] = set()

    try:
        from ptcg_mine.cards import load_engine
        card_data, _attack_data = load_engine()
        for c in card_data:
            cid = c.cardId
            card_type_map[cid] = int(c.cardType)
            if c.aceSpec:
                ace_spec_ids.add(cid)
            if c.basic:
                basic_pokemon_ids.add(cid)
    except Exception:
        logger.warning(
            "Could not load engine card data for deck legality check. "
            "Falling back to heuristic (cardType 5 = basic energy, no ACE SPEC check).",
            exc_info=True,
        )

    return card_type_map, ace_spec_ids, basic_pokemon_ids


def load_fixed_deck(archetypes_path: str | Path) -> list[int] | None:
    """Load FIXED_DECK from archetypes.json (if present).

    Returns ``None`` if the key is absent.
    """
    import json
    path = Path(archetypes_path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("FIXED_DECK")
