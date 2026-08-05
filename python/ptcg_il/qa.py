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

import collections
import logging
from rich.logging import RichHandler
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
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
        The fixed_deck (60 card ids).  If None, deck legality check is skipped.
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
        vl_report = check_variable_length_multi_select(shard_dir, meta)
        results.update(vl_report)
    else:
        results["n_variable_length"] = 0
        results["variable_length_share"] = 0.0

    # --- Attachment-collision audit (D.5.4) ---
    _ac_defaults: dict[str, Any] = {
        "n_collision_options": 0,
        "collision_share": 0.0,
        "n_attachment_options": 0,
        "attachment_samples_checked": 0,
        "n_options_total": 0,
        "n_indistinguishable_options": 0,
        "indistinguishable_share": 0.0,
        "collision_by_opt_type": {},
    }
    if shard_dir.exists() and not meta.empty:
        try:
            ac_report = check_attachment_collision(shard_dir, meta, max_samples=max_samples)
        except ValueError:
            logger.warning("Attachment-collision audit skipped: no options examined.")
            ac_report = _ac_defaults
        results.update(ac_report)
    else:
        results.update(_ac_defaults)

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
    rt_pass, rt_details = check_reference_roundtrip(shard_dir, max_samples=max_samples)
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
    max_samples: int | None = None,
) -> tuple[bool | None, dict[str, Any]]:
    """Verify ``opt_src_idx`` names the state row holding the option's own card.

    The featurizer writes an option's source as a state-token row (A.1) and its
    card features by dereferencing the same location.  Those are two separate
    code paths — ``ref_map.build_ref_map`` for the row, the state-token builders
    for the row's contents — so comparing them catches an off-by-one in either.

    This reads only the shard.  The previous implementation wanted raw
    observations in ``data/observations/``, which nothing has ever written, and
    resolved them with a helper whose index space did not match ``opt_src_idx``
    at all; it could not have passed or failed meaningfully.

    Options whose card row is PAD (all-zero) are skipped: ``card_id_at``
    deliberately leaves deck slots and face-down prizes at PAD, and RETREAT and
    ATTACK point at row 1 without ever setting a card.  Empty state slots are
    PAD for the same reason and are skipped too.

    Returns ``(passed, details)``; ``passed`` is None when nothing was compared.
    """
    details: dict[str, Any] = {
        "n_compared": 0, "n_mismatch": 0,
        "n_src_out_of_range": 0, "n_skipped_pad": 0,
    }
    if shard_dir is None:
        details["skipped"] = "no shard_dir"
        return None, details

    n_compared = n_mismatch = n_oob = n_pad = 0
    examples: list[dict] = []

    for sf in sorted(Path(shard_dir).glob("*.npz")):
        try:
            data = np.load(sf, allow_pickle=False)
        except OSError:
            continue
        needed = ("opt_src_idx", "opt_card_feat", "opt_mask",
                  "poke_card_feat", "hand_card_feat", "stadium_card_feat")
        if any(k not in data for k in needed):
            continue

        opt_src = data["opt_src_idx"]
        opt_card = data["opt_card_feat"]
        opt_mask = data["opt_mask"]
        poke = data["poke_card_feat"]
        hand = data["hand_card_feat"]
        stadium = data["stadium_card_feat"]

        for s in range(opt_src.shape[0]):
            for o in np.flatnonzero(opt_mask[s]):
                row = int(opt_src[s, o])
                if row < 0:
                    continue
                if 1 <= row <= 12:
                    state_row = poke[s, row - 1]
                elif 13 <= row <= 42:
                    state_row = hand[s, row - 13]
                elif row == 45:
                    state_row = stadium[s, 0]
                else:
                    n_oob += 1
                    continue

                card_row = opt_card[s, o]
                if not card_row.any() or not state_row.any():
                    n_pad += 1
                    continue

                n_compared += 1
                if not np.array_equal(card_row, state_row):
                    n_mismatch += 1
                    if len(examples) < 5:
                        examples.append(
                            {"shard": sf.name, "sample": int(s), "option": int(o), "src_row": row}
                        )

            if max_samples is not None and (s + 1) >= max_samples:
                break
        if max_samples is not None:
            break

    details.update(
        n_compared=n_compared, n_mismatch=n_mismatch,
        n_src_out_of_range=n_oob, n_skipped_pad=n_pad, examples=examples,
    )

    if n_oob > 0:
        logger.error(
            "REFERENCE ROUND-TRIP: %d option(s) point at a non-card state row "
            "(CLS/summary/out of range) — the pointer encoding is wrong.", n_oob,
        )
        return False, details

    if n_compared == 0:
        details["skipped"] = (
            "no comparable options — every option was PAD or sourceless, so the "
            "gate examined nothing and must not report a pass."
        )
        logger.info("Reference round-trip QA gate: %s", details["skipped"])
        return None, details

    if n_mismatch > 0:
        logger.error(
            "REFERENCE ROUND-TRIP FAIL: %d/%d options carry card features that do "
            "not match the state row opt_src_idx names.  Examples: %s",
            n_mismatch, n_compared, examples,
        )
        return False, details

    return True, details


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


def _shards_have_stop_column(shard_dir: str | Path) -> bool:
    """Check whether shard files contain the ``stop_column`` key."""
    shard_dir = Path(shard_dir)
    shard_files = sorted(shard_dir.glob("train-*.npz"))
    if not shard_files:
        return False
    try:
        data = np.load(shard_files[0], mmap_mode="r")
        return "stop_column" in data
    except Exception:
        return False


def check_variable_length_multi_select(
    shard_dir: str | Path, meta: pd.DataFrame,
) -> dict[str, Any]:
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
        # Check if shards include STOP column (new featurizer)
        has_stop = _shards_have_stop_column(shard_dir)
        if has_stop:
            logger.info(
                "VARIABLE-LENGTH MULTI-SELECT: %d samples (%.2f%% of multi-select) "
                "have minCount < maxCount.  STOP head IS active — these samples "
                "will receive proper STOP supervision.",
                n_var,
                share * 100,
            )
        else:
            logger.warning(
                "VARIABLE-LENGTH MULTI-SELECT: %d samples (%.2f%% of multi-select) "
                "have minCount < maxCount.  STOP head is in the model but shards "
                "were generated with the old featurizer (no stop_column).  "
                "Regenerate shards to enable STOP supervision.",
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
    ``opt_src_idx`` AND their card's static features (truly indistinguishable to
    the pointer).

    The card half of the comparison is the ``opt_card_feat`` row, not a vocab id:
    the pointer head consumes the feature vector, so two options whose features
    match *are* the same input to it even when the underlying card ids differ.

    Returns ``n_collision_options``, ``collision_share``, ``n_options_total``.
    """
    shard_dir = Path(shard_dir)
    # We only need to check attachment-type options.
    # Walk shard files, reading opt_type, opt_src_idx, opt_card_feat.
    n_collision = 0
    n_total_attachment_options = 0
    samples_checked = 0

    # Group-based pass counters (across ALL option types, not just attachments).
    n_options_seen = 0
    n_indistinguishable = 0
    group_checked = 0
    by_opt_type: dict[int, int] = {}

    shard_files = sorted(shard_dir.glob("*.npz"))
    for sf in shard_files:
        try:
            data = np.load(sf, mmap_mode="r", allow_pickle=False)
        except OSError:
            continue

        # Need at least opt_mask and opt_type for either pass.
        if "opt_type" not in data or "opt_mask" not in data:
            continue
        has_legacy = "opt_card_feat" in data and "opt_src_idx" in data
        has_group = "opt_group" in data
        if not has_legacy and not has_group:
            continue

        opt_type = data["opt_type"]       # [S, O_MAX]
        opt_mask = data["opt_mask"]       # [S, O_MAX]

        # --- Legacy pass: (src, card_feat) key, attachment types only ---
        if has_legacy and (max_samples is None or samples_checked < max_samples):
            opt_src = data["opt_src_idx"]     # [S, O_MAX]
            opt_card = data["opt_card_feat"]  # [S, O_MAX, F_CARD]

            # Attachment OptionTypes: CARD(3), TOOL_CARD(4), ENERGY_CARD(5), ENERGY(6)
            att_mask = np.isin(opt_type, [3, 4, 5, 6]) & opt_mask

            for s in range(opt_type.shape[0]):
                row_mask = att_mask[s]
                if not row_mask.any():
                    continue
                # Build (src_idx, card-features) pairs for this sample's attachment
                # options.  Feature rows are float arrays, so key them by bytes.
                pairs = list(zip(
                    opt_src[s][row_mask].tolist(),
                    [np.ascontiguousarray(row).tobytes()
                     for row in opt_card[s][row_mask]],
                ))
                # Count how many unique pairs vs total
                n_total_attachment_options += len(pairs)
                unique_pairs = set(pairs)
                n_collision += len(pairs) - len(unique_pairs)

                samples_checked += 1
                if max_samples is not None and samples_checked >= max_samples:
                    break

        # --- Group-based pass: true indistinguishability across ALL option types ---
        # True indistinguishability: two options are the same input to the
        # pointer only if *every* tensor it reads agrees.  The legacy
        # (src, card_feat) key above ignores opt_type/opt_tgt_idx/opt_scalar
        # and so overstates the share by ~4x (47.8% vs 10.0% on train-00000).
        if has_group and (max_samples is None or group_checked < max_samples):
            groups = data["opt_group"]
            for s in range(groups.shape[0]):
                row_valid = opt_mask[s]
                if not row_valid.any():
                    continue
                gs = groups[s][row_valid]
                ts = opt_type[s][row_valid]
                counts = collections.Counter(gs.tolist())
                for g_id, t in zip(gs.tolist(), ts.tolist()):
                    n_options_seen += 1
                    if counts[g_id] > 1:
                        n_indistinguishable += 1
                        by_opt_type[int(t)] = by_opt_type.get(int(t), 0) + 1
                group_checked += 1
                if max_samples is not None and group_checked >= max_samples:
                    break

        # Stop only when both passes are done (or no budget was set).
        if max_samples is not None and samples_checked >= max_samples and group_checked >= max_samples:
            break

    if n_options_seen == 0:
        raise ValueError(
            "check_attachment_collision group pass examined no options — "
            "shards lack opt_group (rebuild with updated featurizer) or have "
            "no valid options, so a 0.0 indistinguishable share would be a lie."
        )

    share = n_collision / max(n_total_attachment_options, 1)

    if share > 0.0:
        logger.warning(
            "ATTACHMENT COLLISION (loose upper bound): %d/%d attachment options "
            "(%.2f%%) share opt_src_idx + card features, but this ignores "
            "opt_type/opt_tgt_idx/opt_scalar and overcounts by ~4× vs the "
            "true opt_group number below. Consider promoting attachments to "
            "tokens (A.7).",
            n_collision,
            n_total_attachment_options,
            share * 100,
        )

    report: dict[str, Any] = {
        "n_collision_options": n_collision,
        "collision_share": round(share, 6),
        "n_attachment_options": n_total_attachment_options,
        "attachment_samples_checked": samples_checked,
        "n_options_total": n_options_seen,
        "n_indistinguishable_options": n_indistinguishable,
        "indistinguishable_share": n_indistinguishable / max(n_options_seen, 1),
        "collision_by_opt_type": by_opt_type,
    }

    if report["indistinguishable_share"] > 0.0:
        logger.warning(
            "TRUE OPTION INDISTINGUISHABILITY (opt_group): %d/%d options (%.2f%%) "
            "are indistinguishable from another valid option. "
            "collision_by_opt_type: %s",
            n_indistinguishable,
            n_options_seen,
            report["indistinguishable_share"] * 100,
            by_opt_type,
        )

    return report


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
        # ``_build_label`` appends the STOP pick after the last expert pick, so
        # for those samples action_len == n_expert_picks + 1.  Subtract it back
        # out before range-checking against minCount/maxCount.
        stop_column = (
            data["stop_column"] if "stop_column" in data.keys()
            else np.full(action_idx.shape[0], -1, dtype=np.int64)
        )

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

            # Length — count only the expert picks, not the trailing STOP
            mc = int(min_count[s])
            xc = int(max_count[s])
            n_picks = a_len - (1 if int(stop_column[s]) >= 0 else 0)
            if n_picks < mc or n_picks > xc:
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
    """Load vocab from a JSON file, with int-keyed id maps."""
    import json

    from ptcg_il.featurizer import normalize_vocab

    with open(path) as f:
        return normalize_vocab(json.load(f))


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
    return data.get("fixed_deck")
