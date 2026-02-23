"""
Tests for Feature 3 (Streaming), Feature 4 (Concurrency & Retry),
and Feature 5 (History DB).

Run with:  pytest tests/test_features.py -v
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from d_mmce.observer import Event, EventBus, EventType
from d_mmce.schemas import (
    Critique,
    FinalVerdict,
    ModelResponse,
)

from tests.conftest import FlakyProvider, MockProvider, MockStreamingProvider


# =====================================================================
# Feature 3 — Streaming
# =====================================================================


class TestStreaming:
    def test_token_chunk_event_type_exists(self):
        """TOKEN_CHUNK should be a valid EventType."""
        assert hasattr(EventType, "TOKEN_CHUNK")
        assert EventType.TOKEN_CHUNK.name == "TOKEN_CHUNK"

    def test_mock_streaming_provider_supports_streaming(self):
        p = MockStreamingProvider(name="s1", response_text="Hello")
        assert p.supports_streaming is True

    def test_mock_provider_does_not_support_streaming(self):
        p = MockProvider(name="m1")
        assert p.supports_streaming is False

    @pytest.mark.asyncio
    async def test_generate_stream_calls_on_token(self):
        """generate_stream should invoke on_token callback for each token."""
        p = MockStreamingProvider(name="s1", response_text="ABC")
        tokens_received: list[str] = []

        resp = await p.generate_stream("test", variant="original", on_token=tokens_received.append)

        assert resp.text == "ABC"
        assert tokens_received == ["A", "B", "C"]
        assert resp.provider_name == "s1"
        assert resp.prompt_variant == "original"
        assert resp.latency > 0

    @pytest.mark.asyncio
    async def test_generate_stream_fallback_for_non_streaming(self):
        """Non-streaming providers should still work with generate_stream (fallback)."""
        p = MockProvider(name="m1", response_text="Fallback text")
        tokens: list[str] = []

        resp = await p.generate_stream("test", variant="original", on_token=tokens.append)

        assert resp.text == "Fallback text"
        # Fallback won't produce per-token callbacks
        assert tokens == []

    @pytest.mark.asyncio
    async def test_streaming_emits_token_chunks_in_pipeline(self):
        """When streaming is enabled, the orchestrator should emit TOKEN_CHUNK events."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockStreamingProvider(name="streamer", response_text="Hello world"),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
            enable_streaming=True,
        )

        token_events: list[Event] = []
        engine.event_bus.subscribe(EventType.TOKEN_CHUNK, token_events.append)

        await engine.run("test")

        # Should have received token chunks (at least from the 4 variants)
        assert len(token_events) > 0
        # Each event should have provider, variant, token in payload
        for evt in token_events:
            assert "provider" in evt.payload
            assert "variant" in evt.payload
            assert "token" in evt.payload


# =====================================================================
# Feature 4 — Concurrency & Retry
# =====================================================================


class TestConcurrencyAndRetry:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """With max_concurrent_tasks=2, _parallel_inference should cap concurrency at 2."""
        from d_mmce.orchestrator import D_MMCE

        concurrent_count = 0
        max_concurrent = 0
        call_log: list[int] = []

        class TrackedProvider(MockProvider):
            async def _call(self, prompt):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                peak = concurrent_count
                call_log.append(peak)
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.05)
                concurrent_count -= 1
                return self._response_text, {}

        providers = [
            TrackedProvider(name="t1", response_text="Answer 1"),
            TrackedProvider(name="t2", response_text="Answer 2"),
            TrackedProvider(name="t3", response_text="Answer 3"),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
            max_concurrent_tasks=2,
            enable_streaming=False,
        )

        # Test _parallel_inference in isolation (not full pipeline)
        variants = {"original": "test", "summary": "test summary"}
        concurrent_count = 0
        max_concurrent = 0
        call_log.clear()

        await engine._parallel_inference(providers, variants)

        # 3 providers × 2 variants = 6 tasks, semaphore=2 → peak concurrency ≤ 2
        assert max_concurrent <= 2
        assert len(call_log) == 6  # all tasks completed

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """A flaky provider should succeed after retries."""
        from d_mmce.orchestrator import D_MMCE

        flaky = FlakyProvider(name="flaky", response_text="Eventually works!", fail_count=2)

        engine = D_MMCE(
            providers=[flaky],
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
            max_retries=2,
            retry_base_delay=0.01,  # fast for tests
            enable_streaming=False,
        )

        verdict = await engine.run("test")

        # Should have succeeded (flaky fails 2 times, retries=2 means 3 total attempts)
        assert len(verdict.answer) > 0
        # The provider was called 3 times per variant (2 failures + 1 success)
        # But only for inference tasks — peer review + synthesis also call it
        assert flaky.call_count > 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_still_produces_result(self):
        """If retries are exhausted, the pipeline should handle it gracefully."""
        from d_mmce.orchestrator import D_MMCE

        # Fails 3 times, but only 1 retry allowed → will exhaust
        always_fail = FlakyProvider(name="fail", response_text="Never", fail_count=999)

        engine = D_MMCE(
            providers=[always_fail],
            review_provider_name="auto",
            enable_logging_observer=False,
            max_retries=1,
            retry_base_delay=0.01,
        )

        verdict = await engine.run("test")

        # Pipeline should still return something (error message)
        assert isinstance(verdict, FinalVerdict)
        assert "ERROR" in verdict.answer or verdict.stability_score == 0.0

    @pytest.mark.asyncio
    async def test_default_concurrency_is_four(self):
        """Default max_concurrent_tasks should be 4."""
        from d_mmce.orchestrator import D_MMCE

        engine = D_MMCE(
            providers=[MockProvider()],
            enable_logging_observer=False,
        )
        assert engine._semaphore._value == 4


# =====================================================================
# Feature 5 — History DB
# =====================================================================


class TestHistoryDB:
    @pytest.fixture
    def db(self):
        """Create a history DB backed by a temp file."""
        from d_mmce.history import RunHistoryDB
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_history.db"
            yield RunHistoryDB(db_path=path)

    @pytest.fixture
    def sample_verdict(self):
        return FinalVerdict(
            answer="The answer is 42.",
            stability_score=0.95,
            num_reruns=1,
            audit_trail=["Step 1", "Step 2"],
        )

    @pytest.fixture
    def sample_model_responses(self):
        return [
            ModelResponse(provider_name="model-a", prompt_variant="original", text="Answer A", latency=1.5),
            ModelResponse(provider_name="model-b", prompt_variant="step_by_step", text="Step answer B", latency=2.0),
        ]

    @pytest.fixture
    def sample_critiques(self):
        return [
            Critique(reviewer="model-a", reviewee="model-b", critique_text="VALIDATED", is_validated=True),
            Critique(reviewer="model-b", reviewee="model-a", critique_text="- Issue 1", is_validated=False, issues=["Issue 1"]),
        ]

    @pytest.mark.asyncio
    async def test_save_and_list(self, db, sample_verdict, sample_model_responses):
        run_id = await db.save_run("What is 6*7?", sample_verdict, sample_model_responses)
        assert len(run_id) == 12

        runs = await db.list_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == run_id
        assert runs[0]["query"] == "What is 6*7?"
        assert runs[0]["stability"] == 0.95

    @pytest.mark.asyncio
    async def test_get_run_detail(self, db, sample_verdict, sample_model_responses, sample_critiques):
        run_id = await db.save_run("test query", sample_verdict, sample_model_responses, sample_critiques)

        detail = await db.get_run(run_id)
        assert detail is not None
        assert detail["query"] == "test query"
        assert detail["answer"] == "The answer is 42."
        assert detail["stability"] == 0.95
        assert detail["num_reruns"] == 1
        assert len(detail["responses"]) == 2
        assert len(detail["critiques"]) == 2
        assert detail["audit_trail"] == ["Step 1", "Step 2"]

        # Check critique details
        validated = [c for c in detail["critiques"] if c["is_validated"]]
        assert len(validated) == 1
        not_validated = [c for c in detail["critiques"] if not c["is_validated"]]
        assert not_validated[0]["issues"] == ["Issue 1"]

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, db):
        result = await db.get_run("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_run(self, db, sample_verdict, sample_model_responses):
        run_id = await db.save_run("q", sample_verdict, sample_model_responses)
        assert await db.count_runs() == 1

        deleted = await db.delete_run(run_id)
        assert deleted is True
        assert await db.count_runs() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, db):
        deleted = await db.delete_run("nope")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear_all(self, db, sample_verdict, sample_model_responses):
        await db.save_run("q1", sample_verdict, sample_model_responses)
        await db.save_run("q2", sample_verdict, sample_model_responses)
        assert await db.count_runs() == 2

        count = await db.clear_all()
        assert count == 2
        assert await db.count_runs() == 0

    @pytest.mark.asyncio
    async def test_list_pagination(self, db, sample_verdict, sample_model_responses):
        for i in range(5):
            await db.save_run(f"query {i}", sample_verdict, sample_model_responses)

        page1 = await db.list_runs(limit=2, offset=0)
        assert len(page1) == 2

        page2 = await db.list_runs(limit=2, offset=2)
        assert len(page2) == 2

        page3 = await db.list_runs(limit=2, offset=4)
        assert len(page3) == 1

    @pytest.mark.asyncio
    async def test_ordering_newest_first(self, db, sample_model_responses):
        v1 = FinalVerdict(answer="first", stability_score=0.5)
        v2 = FinalVerdict(answer="second", stability_score=0.9)

        await db.save_run("older", v1, sample_model_responses)
        await asyncio.sleep(0.05)  # ensure different timestamps
        await db.save_run("newer", v2, sample_model_responses)

        runs = await db.list_runs()
        assert runs[0]["query"] == "newer"
        assert runs[1]["query"] == "older"


# =====================================================================
# E2E: Pipeline saves to history
# =====================================================================


class TestPipelineHistoryIntegration:
    @pytest.mark.asyncio
    async def test_pipeline_saves_to_history_db(self):
        """The full pipeline should automatically save to history when db is provided."""
        from d_mmce.history import RunHistoryDB
        from d_mmce.orchestrator import D_MMCE

        with tempfile.TemporaryDirectory() as tmpdir:
            db = RunHistoryDB(db_path=Path(tmpdir) / "test.db")

            providers = [
                MockProvider(name="m1", response_text="Aurora borealis answer."),
            ]
            engine = D_MMCE(
                providers=providers,
                review_provider_name="auto",
                stability_threshold=0.85,
                max_stability_reruns=1,
                enable_logging_observer=False,
                history_db=db,
            )

            verdict = await engine.run("What are the Northern Lights?")

            # Verify it was saved
            assert await db.count_runs() == 1
            runs = await db.list_runs()
            assert runs[0]["query"] == "What are the Northern Lights?"

            # Verify detail
            detail = await db.get_run(runs[0]["id"])
            assert detail["answer"] == verdict.answer
            assert len(detail["responses"]) >= 4  # 1 provider × 4 variants

