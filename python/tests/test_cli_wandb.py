"""Tests for the IL training CLI's W&B wiring.

Runs go to the `poken` team, project `pokemon-tcg-il`.  The bits worth pinning
are the ones that fail *quietly*: an off switch that does not switch anything
off, and several same-named runs from one pipeline invocation.
"""

import argparse
import datetime

import pytest

from ptcg_il import cli


def _parse(*argv):
    return cli._build_parser().parse_args(["train", *argv])


def test_defaults_target_the_poken_team():
    args = _parse()
    assert args.wandb_entity == "poken"
    assert args.wandb_project == "pokemon-tcg-il"
    assert args.wandb_mode == "online"


def test_entity_and_project_are_overridable():
    args = _parse("--wandb-entity", "RBS123", "--wandb-project", "scratch")
    assert args.wandb_entity == "RBS123"
    assert args.wandb_project == "scratch"


class TestExcludeArchBeforeFlag:
    def test_default_is_no_filter(self):
        assert _parse().exclude_arch_before is None

    def test_parses_arch_and_date(self):
        args = _parse("--exclude-arch-before", "1:2026-07-20")
        assert args.exclude_arch_before == (1, datetime.date(2026, 7, 20))

    def test_rejects_malformed_value(self):
        with pytest.raises(SystemExit):
            _parse("--exclude-arch-before", "2026-07-20")


class TestRunName:
    def test_specialist_runs_are_distinguishable(self):
        """Stage 4a trains one model per archetype into the same project."""
        a0 = cli._wandb_run_name(_parse("--archetype-self", "0"))
        a1 = cli._wandb_run_name(_parse("--archetype-self", "1"))
        assert a0 != a1
        assert a0.endswith("-a0") and a1.endswith("-a1")
        assert a0.startswith(cli.DEFAULTS["wandb_name"])

    def test_generalist_is_labelled_too(self):
        assert cli._wandb_run_name(_parse()).endswith("-generalist")

    def test_explicit_name_is_untouched(self):
        args = _parse("--wandb-name", "my-run", "--archetype-self", "3")
        assert cli._wandb_run_name(args) == "my-run"


class TestNoWandb:
    """`--no-wandb` was declared but never read: it used to log online."""

    def test_no_wandb_disables(self):
        assert cli._wandb_mode(_parse("--no-wandb")) == "disabled"

    def test_no_wandb_beats_an_explicit_mode(self):
        args = _parse("--no-wandb", "--wandb-mode", "online")
        assert cli._wandb_mode(args) == "disabled"

    def test_mode_passes_through_without_the_flag(self):
        assert cli._wandb_mode(_parse("--wandb-mode", "offline")) == "offline"
        assert cli._wandb_mode(_parse()) == "online"

    def test_disabled_mode_creates_no_run(self):
        """The consequence that matters: WandbLogger stays inactive, so nothing
        is sent and metrics fall back to the console."""
        from ptcg_il.train.logger import WandbLogger

        log = WandbLogger(project="pokemon-tcg-il", entity="poken", mode="disabled")
        assert log._run is None
        assert not log._active


def test_cmd_train_exports_the_computed_mode(monkeypatch, tmp_path):
    """cmd_train must export it — WandbLogger reads WANDB_MODE, not the args."""
    import os

    monkeypatch.setattr(cli, "_load_artifacts", lambda d: _ARTIFACTS)
    monkeypatch.setattr(cli, "_build_policy", lambda a, args: object())
    # The size gate now wraps the build; a bare object() has no state_dict.
    monkeypatch.setattr(cli, "_model_size_bytes", lambda p: 0)

    import ptcg_il.train.loop as loop

    seen = {}

    def stop(*a, **k):
        seen["mode"] = os.environ.get("WANDB_MODE")
        seen["name"] = k.get("wandb_name")
        seen["entity"] = k.get("wandb_entity")
        raise _Stop

    monkeypatch.setattr(loop, "train", stop)
    monkeypatch.setenv("WANDB_MODE", "online")

    args = _parse("--no-wandb", "--skip-qa", "--archetype-self", "2",
                  "--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "ck"))
    with pytest.raises(_Stop):
        cli.cmd_train(args)

    assert seen["mode"] == "disabled"
    assert seen["entity"] == "poken"
    assert seen["name"].endswith("-a2")


class _Stop(Exception):
    """Ends cmd_train once the W&B wiring has been observed."""


_ARTIFACTS = {
    "vocab": {"size": 4, "id_to_index": {}},
    "vocab_size": 4,
    "attack_size": 2,
    "fixed_deck": [],
    "archetypes": {"self_ids": [0]},
    "card_table": None,
    "attack_table": None,
}
