"""Tests for the submission builder's checkpoint/artifact pairing guard.

The failure being guarded is silent end to end: a checkpoint trained against an
earlier mining run has a vocab of the same *length* but a different id→index
assignment, so the packaged agent loads, imports, plays every game to the end,
and loses nearly all of them without raising anywhere.  `save_checkpoint` pins
`vocab_sha1`/`archetypes_sha1` exactly so this is detectable; the builder used
to print them and compare nothing.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_submission",
    Path(__file__).resolve().parent.parent / "scripts" / "build_submission.py",
)


@pytest.fixture(scope="module")
def bs():
    mod = importlib.util.module_from_spec(_SPEC)
    sys.modules["build_submission"] = mod
    _SPEC.loader.exec_module(mod)
    return mod


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "vocab.json").write_text('{"size": 296}')
    (d / "archetypes.json").write_text('{"self_ids": [0, 1]}')
    return d


def _sha12(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _record(data_dir, **over):
    rec = {
        "archetype_self": 0,
        "vocab_sha1": _sha12(data_dir / "vocab.json"),
        "archetypes_sha1": _sha12(data_dir / "archetypes.json"),
    }
    rec.update(over)
    return rec


def test_matching_pair_is_accepted(bs, data_dir):
    bs.check_artifact_pairing(_record(data_dir), data_dir)


def test_stale_vocab_aborts(bs, data_dir):
    """The real case: checkpoints_a2 pinned d43bb2e2 while data/ held f9470f51."""
    rec = _record(data_dir, vocab_sha1="d43bb2e28112")
    with pytest.raises(SystemExit) as exc:
        bs.check_artifact_pairing(rec, data_dir)
    assert "vocab.json" in str(exc.value)
    assert "d43bb2e28112" in str(exc.value)


def test_stale_archetypes_aborts(bs, data_dir):
    rec = _record(data_dir, archetypes_sha1="c6b5e71baaaa")
    with pytest.raises(SystemExit):
        bs.check_artifact_pairing(rec, data_dir)


def test_force_downgrades_to_a_warning(bs, data_dir, capsys):
    rec = _record(data_dir, vocab_sha1="d43bb2e28112")
    bs.check_artifact_pairing(rec, data_dir, force=True)
    assert "WARNING" in capsys.readouterr().out


def test_truncated_pin_matches_full_digest(bs, data_dir):
    """ptcg_il.deck._sha1 stores 12 chars; comparing against the full 40-char
    digest made every checkpoint look stale, including good ones."""
    full = hashlib.sha1((data_dir / "vocab.json").read_bytes()).hexdigest()
    assert len(full) == 40
    bs.check_artifact_pairing(_record(data_dir, vocab_sha1=full[:12]), data_dir)


def test_unlabelled_checkpoint_is_left_to_the_deck_check(bs, data_dir):
    """build_data_files already refuses a checkpoint with no deck record; this
    guard must not raise a second, more confusing error first."""
    bs.check_artifact_pairing(None, data_dir)


def test_absent_pins_are_not_treated_as_mismatches(bs, data_dir):
    """Checkpoints predating the deck record carry no SHAs to compare."""
    bs.check_artifact_pairing({"archetype_self": 0}, data_dir)
