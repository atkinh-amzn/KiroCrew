"""``POST /v1/chat/completions`` asks for kiro-cli's sign-in only on a kiro-cli harness.

The endpoint refuses a signed-out kiro-cli with 503 ``kiro_prerequisite_required``
when its turn runs on a harness that signs in from kiro-cli's own store. On any
other harness the turn is its own sign-in authority, as for a send. The gate is
asked after the caller is authorized for the slot, so app isolation answers first.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from chat_regenerate_cov80_helpers import state_fixture  # noqa: F401
from chat_test_helpers import pin_harness_choice

from kiro_crew.acp_backends import ACP_BACKEND_CLAUDE, ACP_BACKEND_CODEX, ACP_BACKEND_KIRO
from kiro_crew.config import KiroCrewConfig
from kiro_crew.dashboard.openai_compat import api_completions
from kiro_crew.dashboard.server_internal_paths import _MIXED_INTERNAL_API_PATHS
from kiro_crew.kiro_prerequisite import KiroPrerequisiteService, PrerequisiteStatus


class _SignedOutKiroCli(KiroPrerequisiteService):
    """A kiro-cli that a fresh probe finds signed out."""

    _status = PrerequisiteStatus(platform="Linux")

    async def verified_ready(self, *, max_age_secs: float) -> bool:
        del max_age_secs
        return False


def _request(state, body: dict, app: str = ""):
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    request.app = {"state": state, "kiro_prerequisite_service": object.__new__(_SignedOutKiroCli)}
    request.get = MagicMock(side_effect=lambda k, d="": app if k == "app" else d)
    request.remote = "127.0.0.1"
    request.path = "/v1/chat/completions"
    return request


async def _fake_run_chat(_state, slot, _prompt, **_kwargs):
    slot._pending.append({"role": "assistant", "content": "hi"})
    slot._pending.append({"cls": "done"})
    slot.event.set()


async def _complete(state, body: dict, app: str = ""):
    with patch("kiro_crew.dashboard.openai_compat._run_chat", side_effect=_fake_run_chat):
        resp = await api_completions(_request(state, body, app))
    return resp.status, json.loads(resp.body)


def _pin_applied(state, backend: str) -> None:
    pin_harness_choice(acp_backend=backend)
    state.sessions._cfg = KiroCrewConfig.load()


_MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("named_slot", [False, True])
@pytest.mark.parametrize(
    ("backend", "status"),
    [(ACP_BACKEND_KIRO, 503), (ACP_BACKEND_CLAUDE, 200), (ACP_BACKEND_CODEX, 200)],
)
async def test_a_signed_out_kiro_cli_refuses_only_a_turn_that_runs_on_it(
    state, named_slot, backend, status
) -> None:
    _pin_applied(state, backend)
    body = {"model": "vanellope", "messages": _MESSAGES}
    if named_slot:
        body["id"] = "s1"

    got, payload = await _complete(state, body)

    assert got == status, payload
    if status == 503:
        assert payload["error"]["code"] == "kiro_prerequisite_required"
        assert payload["error"]["type"] == "service_unavailable_error"
        # A refused request leaves no slot behind.
        assert not [k for k in state._slots if k == "s1" or k.startswith("oai-")]
    else:
        assert payload["choices"][0]["message"]["content"] == "hi"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [ACP_BACKEND_KIRO, ACP_BACKEND_CLAUDE])
async def test_app_isolation_answers_before_the_sign_in_gate(state, backend) -> None:
    _pin_applied(state, backend)
    other = state.get_or_create_slot("s1")
    other._app = "other-app"

    got, payload = await _complete(
        state, {"model": "vanellope", "messages": _MESSAGES, "id": "s1"}, app="my-app"
    )
    assert got == 403, payload

    got, payload = await _complete(state, {"model": "vanellope", "messages": _MESSAGES}, "my-app")
    assert got == 403, payload


def test_the_endpoint_keeps_its_caller_auth() -> None:
    assert "/v1/chat/completions" in _MIXED_INTERNAL_API_PATHS


@pytest.mark.asyncio
async def test_a_turn_started_during_the_probe_answers_busy(state) -> None:
    _pin_applied(state, ACP_BACKEND_KIRO)
    slot = state.get_or_create_slot("s1")

    class _ReadyAfterAnotherTurnStarted(KiroPrerequisiteService):
        async def verified_ready(self, *, max_age_secs: float) -> bool:
            del max_age_secs
            slot._turn_admission_reserved = True
            return True

    request = _request(state, {"model": "vanellope", "messages": _MESSAGES, "id": "s1"})
    request.app["kiro_prerequisite_service"] = object.__new__(_ReadyAfterAnotherTurnStarted)
    run_chat = AsyncMock()
    with patch("kiro_crew.dashboard.openai_compat._run_chat", new=run_chat):
        resp = await api_completions(request)

    assert resp.status == 409
    assert json.loads(resp.body)["code"] == "slot_busy"
    run_chat.assert_not_called()


@pytest.mark.asyncio
async def test_a_refusal_keeps_a_new_slot_another_send_took_up_during_the_probe(state) -> None:
    _pin_applied(state, ACP_BACKEND_KIRO)

    class _SignedOutAfterAnotherSend(KiroPrerequisiteService):
        async def verified_ready(self, *, max_age_secs: float) -> bool:
            del max_age_secs
            state._slots["s1"].append("user", "sent from the dashboard")
            return False

    _SignedOutAfterAnotherSend._status = PrerequisiteStatus(platform="Linux")
    request = _request(state, {"model": "vanellope", "messages": _MESSAGES, "id": "s1"})
    request.app["kiro_prerequisite_service"] = object.__new__(_SignedOutAfterAnotherSend)

    resp = await api_completions(request)

    assert resp.status == 503
    assert "s1" in state._slots
