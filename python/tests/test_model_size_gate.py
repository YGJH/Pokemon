"""Model-size gate for IL training (MAX_MODEL_BYTES re-roll).

The shipped weights.pt carries only the state_dict, so the gate measures
exactly that, and it must never silently override an explicitly pinned
architecture.
"""

import argparse

import pytest
import torch

from ptcg_il import cli


def _ns(**kw):
    base = dict(d_model=256, layers=4, heads=2, ff=1024, seed=42,
                attn_dropout=0.0, ffn_dropout=0.0)
    base.update(kw)
    return argparse.Namespace(**base)


class TestModelSizeBytes:
    def test_counts_state_dict_entries(self):
        m = torch.nn.Linear(10, 10)  # (100 + 10) params, fp32
        assert cli._model_size_bytes(m) == 110 * 4

    def test_ignores_nonpersistent_buffers(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(4))
                self.register_buffer("scratch", torch.zeros(1000),
                                     persistent=False)

        assert cli._model_size_bytes(M()) == 4 * 4


class TestExplicitArchFlags:
    def test_none_passed(self):
        assert cli._explicit_arch_flags(["train", "--epochs", "3"]) == set()

    def test_space_and_equals_forms(self):
        got = cli._explicit_arch_flags(["train", "--d-model", "512"])
        assert got == {"d_model"}
        assert cli._explicit_arch_flags(["train", "--layers=5"]) == {"layers"}

    def test_unrelated_subcommand_args_are_ignored(self):
        assert cli._explicit_arch_flags(["build-shards", "--jobs", "4"]) == set()


class TestBuildPolicyUnderCap:
    def test_fits_on_first_draw(self, monkeypatch):
        monkeypatch.setattr(cli, "_build_policy", lambda a, g: object())
        monkeypatch.setattr(cli, "_model_size_bytes", lambda p: 100)
        args = _ns()
        assert cli._build_policy_under_cap({}, args) is not None
        assert args.d_model == 256  # untouched

    def test_rerolls_until_it_fits(self, monkeypatch):
        monkeypatch.setattr(cli, "_build_policy", lambda a, g: object())
        sizes = iter([cli.MAX_MODEL_BYTES + 1] * 2 + [100])
        monkeypatch.setattr(cli, "_model_size_bytes", lambda p: next(sizes))
        draws = [{"d_model": 256, "layers": 4, "heads": 2, "ff": 1024},
                 {"d_model": 512, "layers": 6, "heads": 4, "ff": 2048}]
        monkeypatch.setattr(cli, "_sample_arch", lambda: draws.pop(0))
        args = _ns()
        cli._build_policy_under_cap({}, args)
        assert args.d_model == 512 and args.ff == 2048

    def test_explicit_arch_is_never_rerolled(self, monkeypatch):
        monkeypatch.setattr(cli, "_build_policy", lambda a, g: object())
        monkeypatch.setattr(cli, "_model_size_bytes",
                            lambda p: cli.MAX_MODEL_BYTES + 1)
        args = _ns(_arch_explicit={"d_model"})
        with pytest.raises(SystemExit, match="passed"):
            cli._build_policy_under_cap({}, args)
        assert args.d_model == 256

    def test_gives_up_after_max_draws(self, monkeypatch):
        monkeypatch.setattr(cli, "_build_policy", lambda a, g: object())
        monkeypatch.setattr(cli, "_model_size_bytes",
                            lambda p: cli.MAX_MODEL_BYTES + 1)
        monkeypatch.setattr(cli, "_MAX_ARCH_DRAWS", 3)
        monkeypatch.setattr(cli, "_sample_arch",
                            lambda: {"d_model": 256, "layers": 4,
                                     "heads": 2, "ff": 1024})
        with pytest.raises(SystemExit, match="no sampled architecture"):
            cli._build_policy_under_cap({}, _ns())

    def test_sample_arch_ranges_and_ff_coupling(self):
        for _ in range(50):
            draw = cli._sample_arch()
            assert draw["d_model"] in (256, 512, 1024)
            assert 4 <= draw["layers"] <= 10
            assert draw["heads"] in (2, 4, 8)
            assert draw["ff"] == 4 * draw["d_model"]
            assert draw["d_model"] % draw["heads"] == 0
