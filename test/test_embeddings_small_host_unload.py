"""A small machine releases the embedding model sooner, and nobody waits for it.

On a machine with 16 GiB of memory or less the idle unload defaults to ten
minutes instead of an hour. Opening, focusing or typing in a chat loads the
model again (or keeps it loaded), so a person's first message finds it there.
The evictor never unloads under an embed in flight or a background sweep that
is pausing between texts. A larger machine keeps every default it had, and an
operator's own value wins everywhere.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import embeddings_helpers
import pytest
from embeddings_helpers import _make_fake_llama_class, _write_model_file

import kiro_crew.embeddings as emb
import kiro_crew.embeddings_settings as settings
import kiro_crew.embeddings_shared as shared
from kiro_crew.embeddings import LlamaCppEmbedder
from kiro_crew.testing.wait import FALLBACK_TIMEOUT_SECS, wait_until

_reset_embedding_singletons = embeddings_helpers._reset_embedding_singletons


@pytest.fixture
def host(monkeypatch):
    """Set whether this machine is small, for the code under test only."""

    def _set(small: bool) -> None:
        monkeypatch.setattr(emb, "small_memory_host", lambda: small)

    return _set


@pytest.mark.parametrize(
    ("total_mib", "small"),
    [(8 * 1024, True), (16 * 1024, True), (16 * 1024 + 1, False), (32 * 1024, False), (0, False)],
)
def test_small_means_16_gib_or_less_and_an_unread_size_is_not_small(monkeypatch, total_mib, small):
    import kiro_crew.platform_compat as platform_compat

    monkeypatch.setattr(platform_compat, "host_total_mib", lambda: total_mib)
    monkeypatch.setattr(settings, "_small_host_cache", [])
    assert settings.small_memory_host() is small


@pytest.mark.parametrize(
    ("raw", "small", "expected"),
    [
        (None, True, 600.0),
        (3600, True, 600.0),  # the saved field default is default policy
        (3599, True, 3599.0),
        (0, True, 0.0),
        (120, True, 120.0),
        (None, False, 3600.0),
        (3600, False, 3600.0),
        (0, False, 0.0),
        (120, False, 120.0),
        ("soon", False, 3600.0),
    ],
)
def test_the_idle_unload_default_depends_on_the_machine_and_a_set_value_wins(
    monkeypatch, host, raw, small, expected
):
    host(small)
    section = {} if raw is None else {"embedding_idle_evict_secs": raw}
    monkeypatch.setattr(emb, "_read_memory_config", lambda: section)
    assert emb.embedding_idle_evict_secs() == expected


class TestEvictionWaitsForWork:
    def _loaded(self, tmp_path: Path, monkeypatch) -> LlamaCppEmbedder:
        fake_cls = _make_fake_llama_class()
        fake_cls.close = lambda self: None
        monkeypatch.setattr("kiro_crew.embeddings._load_llama_class", lambda: fake_cls)
        backend = LlamaCppEmbedder(model_path=_write_model_file(tmp_path / "model.gguf"))
        assert backend.embed("warm") is not None
        # Join the loader: evict_if_idle refuses while it is alive (its epilogue
        # starts the evictor after the model is published).
        assert backend.wait_ready(timeout=FALLBACK_TIMEOUT_SECS)
        assert backend._load_thread is not None and not backend._load_thread.is_alive()
        backend._last_used = time.monotonic() - 10_000
        return backend

    def test_an_embed_in_flight_holds_the_model(self, tmp_path, monkeypatch):
        backend = self._loaded(tmp_path, monkeypatch)
        try:
            entered, release = threading.Event(), threading.Event()
            real = backend._embed_batch

            def held(texts, *, priority):
                entered.set()
                assert release.wait(FALLBACK_TIMEOUT_SECS)
                return real(texts, priority=priority)

            backend._embed_batch = held  # type: ignore[method-assign]
            worker = threading.Thread(target=backend.embed, args=("a recall",), daemon=True)
            worker.start()
            assert entered.wait(FALLBACK_TIMEOUT_SECS)
            backend._last_used = time.monotonic() - 10_000
            assert backend.evict_if_idle(1.0) is False
            assert backend.is_ready()
            release.set()
            worker.join(FALLBACK_TIMEOUT_SECS)
            assert not worker.is_alive()
            backend._last_used = time.monotonic() - 10_000
            assert backend.evict_if_idle(1.0) is True
        finally:
            backend.close()

    def test_a_background_sweep_between_texts_holds_the_model(self, tmp_path, monkeypatch):
        backend = self._loaded(tmp_path, monkeypatch)
        try:
            backend._bulk_ready_at = time.monotonic() + 3600
            assert backend.evict_if_idle(1.0) is False
            backend._bulk_ready_at = 0.0
            assert backend.evict_if_idle(1.0) is True
        finally:
            backend.close()


class TestKeepWarm:
    def test_a_loaded_model_starts_its_idle_clock_again(self, tmp_path, monkeypatch):
        backend = TestEvictionWaitsForWork()._loaded(tmp_path, monkeypatch)
        try:
            before = time.monotonic()
            backend.keep_warm()
            assert backend._last_used >= before
            assert backend.evict_if_idle(60.0) is False
        finally:
            backend.close()

    def test_an_unloaded_model_is_loaded_before_anyone_embeds(self, tmp_path, monkeypatch):
        backend = TestEvictionWaitsForWork()._loaded(tmp_path, monkeypatch)
        try:
            assert backend.evict_if_idle(1.0) is True
            assert not backend.is_ready()
            backend.keep_warm()
            wait_until(
                backend.is_ready,
                timeout=FALLBACK_TIMEOUT_SECS,
                describe=lambda: "the model loads again",
            )
        finally:
            backend.close()


class TestWarmForPerson:
    def _install(self, monkeypatch) -> MagicMock:
        backend = MagicMock()
        monkeypatch.setattr(shared, "_shared_embedder", backend)
        return backend

    def test_a_small_machine_warms_the_embedder_it_already_has(self, monkeypatch, host):
        host(True)
        backend = self._install(monkeypatch)
        emb.warm_embedder_for_person()
        backend.keep_warm.assert_called_once_with()

    def test_a_larger_machine_does_nothing(self, monkeypatch, host):
        host(False)
        backend = self._install(monkeypatch)
        emb.warm_embedder_for_person()
        backend.keep_warm.assert_not_called()

    def test_no_embedder_is_built_for_a_focus(self, monkeypatch, host):
        host(True)
        monkeypatch.setattr(shared, "_shared_embedder", None)
        built = MagicMock()
        monkeypatch.setattr(shared, "default_embedding_backend", built)
        emb.warm_embedder_for_person()
        built.assert_not_called()

    @pytest.mark.asyncio
    async def test_from_the_loop_it_runs_on_the_executor(self, monkeypatch, host):
        host(True)
        loop_thread = threading.get_ident()
        ran_on: list[int] = []
        done = asyncio.Event()
        loop = asyncio.get_running_loop()

        def warm() -> None:
            ran_on.append(threading.get_ident())
            loop.call_soon_threadsafe(done.set)

        monkeypatch.setattr(shared, "warm_embedder_for_person", warm)
        emb.warm_embedder_for_person_soon()
        await asyncio.wait_for(done.wait(), FALLBACK_TIMEOUT_SECS)
        assert ran_on and ran_on[0] != loop_thread

    def test_the_owners_focus_frame_asks_for_it(self, monkeypatch):
        from kiro_crew.dashboard import chat_runner
        from kiro_crew.dashboard.state import _ChatSlot
        from kiro_crew.dashboard.ws import _handle_slot_focused

        asked = MagicMock()
        monkeypatch.setattr(shared, "warm_embedder_for_person_soon", asked)
        monkeypatch.setattr(chat_runner, "schedule_eager_spawn", MagicMock())
        state = MagicMock()
        state.get_slot = MagicMock(return_value=_ChatSlot("t1"))
        _handle_slot_focused(state, "t1", None, owner=True)
        asked.assert_called_once_with()
        asked.reset_mock()
        _handle_slot_focused(state, "t1", None, owner=False)
        asked.assert_not_called()
