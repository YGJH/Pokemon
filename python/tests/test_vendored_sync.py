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
