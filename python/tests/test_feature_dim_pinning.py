"""A checkpoint must record the featurizer widths it was trained with.

Feature widths are module-level constants baked into layer shapes at
construction (``cards.py`` ``MLP(F_CARD, D, D)``, ``embed.py`` ``MLP(F_POKE,
…)``).  Editing ``featurizer.py`` changes every one of them, and nothing in
``Policy.config`` recorded them — so a checkpoint and a shard set built from
different featurizer generations were only distinguishable by a
``load_state_dict`` size mismatch, whose message names layer shapes rather than
the actual cause.  This is the same rationale as the ``vocab_sha1`` /
``archetypes_sha1`` pinning in ``ptcg_il.deck``.

Mutation check: dropping ``feat_dims`` from ``Policy.config`` fails
``test_config_records_feature_widths``; removing the comparison in
``policy_from_config`` fails ``test_policy_from_config_rejects_stale_widths``.
"""

import pytest
import torch

from ptcg_il import featurizer as fz
from ptcg_il.model.policy import FEATURE_DIM_KEYS, Policy, policy_from_config


def _config():
    return Policy(D=32, heads=2, layers=1, ff=64).config


def test_config_records_feature_widths():
    cfg = _config()
    assert "feat_dims" in cfg, "Policy.config must record the featurizer widths"
    dims = cfg["feat_dims"]
    assert set(dims) == set(FEATURE_DIM_KEYS), (
        f"feat_dims keys {sorted(dims)} != {sorted(FEATURE_DIM_KEYS)}"
    )
    assert dims, "degenerate: FEATURE_DIM_KEYS is empty"
    for k in FEATURE_DIM_KEYS:
        assert dims[k] == getattr(fz, k), f"{k} recorded as {dims[k]}, live {getattr(fz, k)}"


def test_policy_from_config_accepts_matching_widths():
    pol = policy_from_config(_config())
    assert pol.config["feat_dims"] == _config()["feat_dims"]


def test_policy_from_config_rejects_stale_widths():
    """The pre-2026-08-04 featurizer had F_CARD=94; rebuilding must raise."""
    cfg = _config()
    cfg["feat_dims"] = dict(cfg["feat_dims"])
    cfg["feat_dims"]["F_CARD"] = 94
    with pytest.raises(ValueError, match="F_CARD"):
        policy_from_config(cfg)


def test_error_names_every_mismatched_width():
    cfg = _config()
    cfg["feat_dims"] = dict(cfg["feat_dims"])
    cfg["feat_dims"]["F_CARD"] = 94
    cfg["feat_dims"]["F_HAND"] = 2
    with pytest.raises(ValueError) as exc:
        policy_from_config(cfg)
    msg = str(exc.value)
    assert "F_CARD" in msg and "F_HAND" in msg
    assert "94" in msg and str(fz.F_CARD) in msg


def test_config_without_feat_dims_is_tolerated():
    """Checkpoints written before this record exist; they must still rebuild."""
    cfg = _config()
    cfg.pop("feat_dims")
    pol = policy_from_config(cfg)
    assert pol.config["feat_dims"]


def test_saved_checkpoint_carries_feature_widths(tmp_path):
    from ptcg_il.train.checkpoint import save_checkpoint

    pol = Policy(D=32, heads=2, layers=1, ff=64)
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)

    class _EMAStub:
        def state_dict(self):
            return {}

    path = save_checkpoint(
        pol, opt, sched, _EMAStub(), step=1, save_dir=tmp_path,
        deck={"deck": [1] * 60, "archetype_self": 0},
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["config"]["feat_dims"]["F_CARD"] == fz.F_CARD
