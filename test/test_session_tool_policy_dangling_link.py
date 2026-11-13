"""A link to a missing file in the agents directory refuses only its own agent.

Real files, a real dangling symlink and the real policy read: nothing is mocked.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from kiro_crew.dashboard.handlers import sessions as sessions_mod
from kiro_crew.dashboard.handlers.sessions_parts import tool_policy

_read = sessions_mod._read_managed_tool_policy_sync


@pytest.fixture
def agents_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tool_policy, "_UNREADABLE_SPEC_WARNED", set())
    (tmp_path / "good.json").write_text(
        json.dumps({"name": "good", "managedToolPolicy": {"exclude": ["x"]}})
    )
    os.symlink(tmp_path / "missing-target.json", tmp_path / "broken-link.json")
    return tmp_path


def test_other_agents_still_resolve(agents_dir: Path) -> None:
    assert _read(agents_dir, "good") == {"exclude": ["x"]}
    assert _read(agents_dir, "no-spec-agent") is None


def test_the_dangling_link_is_logged_once_with_its_path(
    agents_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=tool_policy.logger.name):
        for _ in range(3):
            _read(agents_dir, "no-spec-agent")
            _read(agents_dir, "other-agent")
    named = [r for r in caplog.records if "broken-link.json" in r.getMessage()]
    assert len(named) == 1


def test_the_agent_whose_own_spec_is_the_dangling_link_stays_refused(agents_dir: Path) -> None:
    with pytest.raises(sessions_mod.ManagedToolPolicyUnreadable):
        _read(agents_dir, "broken-link")


def test_a_link_to_an_unparseable_file_still_refuses_every_spec_less_agent(
    agents_dir: Path,
) -> None:
    (agents_dir / "garbled-target.txt").write_text("{not json")
    os.symlink(agents_dir / "garbled-target.txt", agents_dir / "alias.json")
    with pytest.raises(sessions_mod.ManagedToolPolicyUnreadable):
        _read(agents_dir, "no-spec-agent")


def test_a_broken_regular_spec_still_refuses_every_spec_less_agent(agents_dir: Path) -> None:
    (agents_dir / "garbled.json").write_text("{not json")
    with pytest.raises(sessions_mod.ManagedToolPolicyUnreadable):
        _read(agents_dir, "no-spec-agent")


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs a directory the current user cannot traverse",
)
def test_a_link_into_a_directory_without_traversal_still_refuses(agents_dir: Path) -> None:
    locked = agents_dir / "locked"
    locked.mkdir()
    (locked / "target.json").write_text(json.dumps({"name": "hidden"}))
    os.symlink(locked / "target.json", agents_dir / "hidden-link.json")
    locked.chmod(0o000)
    try:
        with pytest.raises(sessions_mod.ManagedToolPolicyUnreadable) as refused:
            _read(agents_dir, "no-spec-agent")
        assert "hidden-link.json" in str(refused.value)
    finally:
        locked.chmod(0o700)


def test_a_second_name_for_the_agents_own_dangling_link_is_refused(agents_dir: Path) -> None:
    os.symlink(agents_dir / "missing-target.json", agents_dir / "helper.json")
    try:
        os.link(agents_dir / "helper.json", agents_dir / "Alias.json", follow_symlinks=False)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"this filesystem cannot hard-link a symlink: {exc}")
    with pytest.raises(sessions_mod.ManagedToolPolicyUnreadable) as refused:
        _read(agents_dir, "helper")
    assert "Alias.json" in str(refused.value)


def test_a_dangling_link_in_another_case_than_the_agents_own_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool_policy, "_UNREADABLE_SPEC_WARNED", set())
    (tmp_path / "Probe.json").write_text("{}")
    if not (tmp_path / "probe.json").exists():
        pytest.skip("this filesystem is case-sensitive")
    os.symlink(tmp_path / "missing-target.json", tmp_path / "Helper.json")
    with pytest.raises(sessions_mod.ManagedToolPolicyUnreadable):
        _read(tmp_path, "helper")
