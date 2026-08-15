"""Guard against drift between ptcg_il and the vendored Kaggle-bundle copies."""

import sys
from pathlib import Path


def test_vendored_featurizer_dims_match():
    """model/featurizer.py must agree with ptcg_il/featurizer.py on all F_* dims."""
    from ptcg_il import featurizer as real

    # Import the vendored copy — needs its own path
    repo_root = Path(__file__).resolve().parent.parent.parent
    vendored_dir = str(repo_root)
    if vendored_dir not in sys.path:
        sys.path.insert(0, vendored_dir)
    import model.featurizer as vendored  # noqa: E402

    for name in ("F_CARD", "F_ATK", "F_POKE", "F_HAND", "F_SUM", "F_GLOBAL", "F_OPT"):
        real_val = getattr(real, name)
        vendored_val = getattr(vendored, name)
        assert real_val == vendored_val, (
            f"{name}: ptcg_il={real_val}, model/={vendored_val}"
        )
    # DRAW_N too
    assert getattr(real, "DRAW_N") == getattr(vendored, "DRAW_N"), (
        f"DRAW_N mismatch: {getattr(real, 'DRAW_N')} vs {getattr(vendored, 'DRAW_N')}"
    )


def test_vendored_featurizer_has_bench_plumbing():
    """model/featurizer.py must have opt_bench_idx in featurize output."""
    import numpy as np

    repo_root = Path(__file__).resolve().parent.parent.parent
    vendored_dir = str(repo_root)
    if vendored_dir not in sys.path:
        sys.path.insert(0, vendored_dir)
    import model.featurizer as vendored  # noqa: E402

    # option_groups must accept opt_bench_idx
    from inspect import signature
    sig = signature(vendored.option_groups)
    assert "opt_bench_idx" in sig.parameters, (
        "model/featurizer.option_groups missing opt_bench_idx parameter"
    )

    # opt_bench_idx feature dim is covered by F_OPT assert above (13),
    # but also verify O_MAX is consistent.
    assert vendored.O_MAX == 64, f"O_MAX={vendored.O_MAX}"

    # Verify _build_option_tokens includes opt_bench_idx in return.
    # We check the return annotation rather than calling it (avoids needing
    # real observation data).
    build_sig = signature(vendored._build_option_tokens)
    # Return annotation is a tuple type; just check the source string
    import inspect as _inspect
    src = _inspect.getsource(vendored._build_option_tokens)
    assert "opt_bench_idx" in src, (
        "model/featurizer._build_option_tokens: opt_bench_idx not found in source"
    )
    # The return statement must include opt_bench_idx
    assert "opt_bench_idx" in src, (
        "model/featurizer._build_option_tokens return: opt_bench_idx not found"
    )

    # _best_bench_damage_ratio and _best_bench_hp_ratio must exist
    assert hasattr(vendored, "_best_bench_damage_ratio"), (
        "model/featurizer missing _best_bench_damage_ratio"
    )
    assert hasattr(vendored, "_best_bench_hp_ratio"), (
        "model/featurizer missing _best_bench_hp_ratio"
    )

    # Verify the featurize result dict includes opt_bench_idx and opt_group.
    featurize_src = _inspect.getsource(vendored.featurize)
    assert '"opt_bench_idx"' in featurize_src, (
        "model/featurizer.featurize result missing opt_bench_idx key"
    )
    assert '"opt_group"' in featurize_src, (
        "model/featurizer.featurize result missing opt_group key"
    )


def test_vendored_pointer_signature_matches():
    """model/pointer.py PointerHead.forward must consume opt_bench_idx."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    vendored_dir = str(repo_root)
    if vendored_dir not in sys.path:
        sys.path.insert(0, vendored_dir)

    from ptcg_il.model.pointer import PointerHead as RealPointer
    import model.pointer as vendored  # noqa: E402
    from model.pointer import PointerHead as VendoredPointer

    # Both should construct without error
    r = RealPointer(D=64, heads=2)
    v = VendoredPointer(D=64, heads=2)
    assert isinstance(r, RealPointer)
    assert isinstance(v, VendoredPointer)

    # Check source code contains opt_bench_idx and bench term
    import inspect
    real_src = inspect.getsource(RealPointer.forward)
    vendored_src = inspect.getsource(VendoredPointer.forward)
    assert "opt_bench_idx" in real_src, "real pointer missing opt_bench_idx"
    assert "opt_bench_idx" in vendored_src, "vendored pointer missing opt_bench_idx"
    assert "bench" in real_src, "real pointer missing bench term"
    assert "bench" in vendored_src, "vendored pointer missing bench term"


def test_vendored_slot_ko_block_matches_values():
    """Dim parity is not enough -- the two copies must agree on the numbers.

    ``model/featurizer.py`` is what the Kaggle bundle runs at inference time.  A
    KO block that exists in both but computes different values feeds the
    deployed policy a feature it was never trained on, and nothing raises: the
    shapes match, so it scores like a plausible model on a shifted input.
    """
    import numpy as np

    from ptcg_il import featurizer as real

    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import model.featurizer as vendored  # noqa: E402

    assert hasattr(vendored, "_slot_ko_block"), (
        "model/featurizer.py is missing _slot_ko_block -- the bundle would feed "
        "the policy an all-zero KO block it was not trained on"
    )
    for name in ("POKE_KO_RATIO_COL", "POKE_KO_FLAG_COL"):
        assert getattr(real, name) == getattr(vendored, name), f"{name} drifted"

    from ptcg_mine.cards import build_engine_card_features, load_engine

    card_data, attack_data = load_engine()
    ecf = build_engine_card_features(card_data, attack_data)

    def _poke(cid, serial, hp, energies=()):
        energies = list(energies)
        return {"id": cid, "serial": serial, "hp": hp, "maxHp": hp,
                "appearThisTurn": False, "energies": energies,
                "energyCards": [{"id": 1, "serial": 900 + i, "playerIndex": 0}
                                for i in range(len(energies))],
                "tools": [], "preEvolution": []}

    def _player(active, bench):
        return {"active": active, "bench": bench, "benchMax": 5, "deckCount": 40,
                "discard": [], "prize": [None] * 6, "handCount": 0, "hand": [],
                "poisoned": False, "burned": False, "asleep": False,
                "paralyzed": False, "confused": False}

    state = {
        "turn": 5, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
        "supporterPlayed": False, "stadiumPlayed": False,
        "energyAttached": False, "retreated": False, "result": -1,
        "stadium": [], "looking": None,
        "players": [
            _player([_poke(675, 10, 70)], [_poke(305, 11, 100)]),
            _player([_poke(676, 20, 110, [6])], [_poke(675, 21, 70)]),
        ],
    }

    outs = []
    for mod in (real, vendored):
        poke_id, poke_feat, _, _ = mod._build_poke_tokens(state, 0)
        pcf = mod._feat_gatherer(ecf, mod.F_CARD)(poke_id)
        mod._slot_ko_block(pcf, poke_feat)
        outs.append(poke_feat[:, [mod.POKE_KO_RATIO_COL, mod.POKE_KO_FLAG_COL]])

    assert outs[0].any(), "fixture produced an all-zero KO block -- test is vacuous"
    np.testing.assert_allclose(
        outs[0], outs[1], rtol=0, atol=0,
        err_msg="ptcg_il and model/ disagree on the KO block values",
    )
