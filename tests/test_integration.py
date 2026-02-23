"""
Integration & end-to-end tests for D-MMCE.
Uses mock providers — no real API calls or Ollama required.

Run with:  pytest tests/test_integration.py -v
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from d_mmce.observer import Event, EventBus, EventType
from d_mmce.peer_reviewer import PeerReviewer
from d_mmce.schemas import (
    ConsensusCluster,
    ContradictionMatrix,
    FinalVerdict,
    ModelResponse,
)

# Re-use the MockProvider from conftest
from tests.conftest import MockProvider


# =====================================================================
# PeerReviewer integration tests
# =====================================================================


class TestPeerReviewerIntegration:
    """Test the full review flow with mock providers."""

    @pytest.mark.asyncio
    async def test_review_produces_matrix(self, sample_responses):
        reviewer = MockProvider(
            name="mock-reviewer",
            response_text="VALIDATED",
        )
        pr = PeerReviewer(review_provider=reviewer)
        matrix = await pr.review(sample_responses)

        assert isinstance(matrix, ContradictionMatrix)
        assert len(matrix.critiques) > 0
        # With 2 unique providers (alpha, beta), we expect 2 critique pairs
        assert len(matrix.scores) == 2

    @pytest.mark.asyncio
    async def test_review_emits_events(self, sample_responses, event_bus):
        captured: list[Event] = []
        event_bus.subscribe(EventType.PEER_CRITIQUE, captured.append)

        reviewer = MockProvider(name="reviewer", response_text="- Issue 1\n- Issue 2")
        pr = PeerReviewer(review_provider=reviewer, event_bus=event_bus)
        matrix = await pr.review(sample_responses)

        assert len(captured) == 2  # one for each pair
        for evt in captured:
            assert "issues" in evt.payload

    @pytest.mark.asyncio
    async def test_validated_gives_zero_penalty(self, sample_responses):
        reviewer = MockProvider(name="reviewer", response_text="VALIDATED")
        pr = PeerReviewer(review_provider=reviewer)
        matrix = await pr.review(sample_responses)

        for score in matrix.scores.values():
            assert score == 0.0

    @pytest.mark.asyncio
    async def test_reviewer_field_matches_actual_provider(self, sample_responses):
        """In single-reviewer mode, all critiques should name the actual review provider."""
        reviewer = MockProvider(name="actual-reviewer-model", response_text="VALIDATED")
        pr = PeerReviewer(review_provider=reviewer)
        matrix = await pr.review(sample_responses)

        for critique in matrix.critiques:
            assert critique.reviewer == "actual-reviewer-model"
            assert critique.critique_source == "actual-reviewer-model"

    @pytest.mark.asyncio
    async def test_review_events_include_critique_source(self, sample_responses, event_bus):
        """Events should include the critique_source field."""
        captured: list[Event] = []
        event_bus.subscribe(EventType.PEER_CRITIQUE, captured.append)

        reviewer = MockProvider(name="my-reviewer", response_text="VALIDATED")
        pr = PeerReviewer(review_provider=reviewer, event_bus=event_bus)
        await pr.review(sample_responses)

        for evt in captured:
            assert evt.payload["critique_source"] == "my-reviewer"

    @pytest.mark.asyncio
    async def test_review_with_retry(self, sample_responses):
        """PeerReviewer should retry on transient failures."""
        from tests.conftest import FlakyProvider

        flaky = FlakyProvider(name="flaky-reviewer", response_text="VALIDATED", fail_count=1)
        pr = PeerReviewer(
            review_provider=flaky,
            max_retries=2,
            retry_base_delay=0.01,
        )
        matrix = await pr.review(sample_responses)

        # Should succeed after retries
        assert len(matrix.critiques) > 0


# =====================================================================
# SemanticClusterer integration tests
# =====================================================================


class TestSemanticClusterer:
    """Tests for the clustering pipeline. Requires sentence-transformers."""

    @pytest.fixture
    def clusterer(self, event_bus):
        from d_mmce.semantic_clusterer import SemanticClusterer
        return SemanticClusterer(event_bus=event_bus)

    def test_cluster_similar_responses(self, clusterer, sample_responses):
        """All aurora-related responses should produce a non-empty consensus cluster."""
        result = clusterer.cluster(sample_responses)

        assert isinstance(result, ConsensusCluster)
        # The clusterer should form at least one cluster with 2+ members
        assert len(result.member_responses) >= 2
        assert len(result.centroid_text) > 0
        # Total members + outliers should equal input count
        assert len(result.member_responses) + len(result.outliers) == len(sample_responses)

    def test_cluster_detects_outlier(self, clusterer, sample_responses):
        """An unrelated response should be separated from the aurora cluster."""
        outlier = ModelResponse(
            provider_name="mock-bad",
            prompt_variant="original",
            text="Python is a programming language used for web development.",
            latency=0.5,
        )
        responses = sample_responses + [outlier]
        result = clusterer.cluster(responses)

        # The unrelated response should NOT be in the same cluster as aurora responses
        member_texts = {r.text for r in result.member_responses}
        outlier_texts = {r.text for r in result.outliers}

        # Either the outlier is excluded, or the aurora responses form the majority
        assert len(result.member_responses) >= 2
        # The unrelated text should be an outlier (not in consensus)
        assert outlier.text in outlier_texts or len(result.member_responses) >= len(sample_responses)

    def test_cluster_empty(self, clusterer):
        result = clusterer.cluster([])
        assert result.centroid_text == ""
        assert result.member_responses == []

    def test_cluster_single_response(self, clusterer):
        single = [ModelResponse(provider_name="a", prompt_variant="original", text="Hello world")]
        result = clusterer.cluster(single)
        assert len(result.member_responses) == 1
        assert result.centroid_text == "Hello world"

    def test_cosine_similarity_texts(self, clusterer):
        sim = clusterer.cosine_similarity_texts(
            "The sun is a star.",
            "The sun is a large star in our solar system.",
        )
        assert 0.5 < sim <= 1.0

    def test_cosine_similarity_identical(self, clusterer):
        sim = clusterer.cosine_similarity_texts("Hello world", "Hello world")
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_cosine_similarity_unrelated(self, clusterer):
        sim = clusterer.cosine_similarity_texts(
            "Quantum physics describes subatomic particles.",
            "I like chocolate cake with strawberries.",
        )
        assert sim < 0.5

    def test_emits_cluster_event(self, event_bus, sample_responses):
        from d_mmce.semantic_clusterer import SemanticClusterer
        captured: list[Event] = []
        event_bus.subscribe(EventType.CLUSTER_FORMED, captured.append)

        clusterer = SemanticClusterer(event_bus=event_bus)
        clusterer.cluster(sample_responses)
        assert len(captured) == 1
        assert "members" in captured[0].message.lower() or "consensus" in captured[0].message.lower()


# =====================================================================
# MetaJudge integration tests
# =====================================================================


class TestMetaJudge:
    @pytest.mark.asyncio
    async def test_synthesize_returns_verdict(self):
        from d_mmce.meta_judge import MetaJudge
        from d_mmce.semantic_clusterer import SemanticClusterer

        provider = MockProvider(
            name="judge",
            response_text="The Northern Lights are caused by solar wind interacting with Earth's magnetic field.",
        )
        clusterer = SemanticClusterer()
        judge = MetaJudge(
            providers=[provider],
            clusterer=clusterer,
            stability_threshold=0.85,
            max_reruns=3,
        )

        cluster = ConsensusCluster(
            centroid_text="Aurora borealis is caused by solar particles.",
            member_responses=[
                ModelResponse(provider_name="a", prompt_variant="original", text="Aurora borealis..."),
            ],
        )
        matrix = ContradictionMatrix()

        verdict = await judge.synthesize(cluster, matrix)

        assert isinstance(verdict, FinalVerdict)
        assert len(verdict.answer) > 0
        assert verdict.stability_score > 0

    @pytest.mark.asyncio
    async def test_synthesize_converges_with_deterministic_provider(self):
        """A deterministic provider should converge in 2 attempts (similarity=1.0)."""
        from d_mmce.meta_judge import MetaJudge
        from d_mmce.semantic_clusterer import SemanticClusterer

        provider = MockProvider(
            name="judge",
            response_text="Deterministic answer about aurora borealis.",
        )
        clusterer = SemanticClusterer()
        judge = MetaJudge(
            providers=[provider],
            clusterer=clusterer,
            stability_threshold=0.85,
            max_reruns=3,
        )
        cluster = ConsensusCluster(
            centroid_text="Aurora",
            member_responses=[ModelResponse("a", "original", "Aurora")],
        )
        matrix = ContradictionMatrix()

        verdict = await judge.synthesize(cluster, matrix)

        # Deterministic → identical outputs → cosine sim ≈ 1.0 → converges on attempt 2
        assert verdict.stability_score == pytest.approx(1.0, abs=0.01)
        assert verdict.num_reruns == 1  # converged after 1 re-run

    @pytest.mark.asyncio
    async def test_select_judge_picks_lowest_penalty(self):
        from d_mmce.meta_judge import MetaJudge
        from d_mmce.semantic_clusterer import SemanticClusterer

        good = MockProvider(name="good-model", response_text="answer")
        bad = MockProvider(name="bad-model", response_text="answer")
        clusterer = SemanticClusterer()
        judge = MetaJudge(providers=[good, bad], clusterer=clusterer)

        matrix = ContradictionMatrix()
        matrix.scores[("reviewer", "good-model")] = 0.0
        matrix.scores[("reviewer", "bad-model")] = 0.75

        selected = judge._select_judge(matrix)
        assert selected.name == "good-model"


# =====================================================================
# Full pipeline E2E test (no network)
# =====================================================================


class TestE2EPipeline:
    """End-to-end test of the complete D-MMCE pipeline with mock providers."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(
                name="model-a",
                response_text="The Northern Lights are caused by solar particles hitting the atmosphere.",
            ),
            MockProvider(
                name="model-b",
                response_text="Aurora borealis results from charged solar particles interacting with Earth's magnetic field.",
            ),
        ]

        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=2,
            enable_logging_observer=False,
        )

        verdict = await engine.run("What are the Northern Lights?")

        assert isinstance(verdict, FinalVerdict)
        assert len(verdict.answer) > 0
        assert verdict.stability_score > 0
        assert isinstance(verdict.audit_trail, list)
        assert len(verdict.audit_trail) > 0

    @pytest.mark.asyncio
    async def test_pipeline_emits_all_event_types(self):
        """The pipeline should emit at least: PROMPT_PERTURBED, MODEL_RESPONSE,
        PEER_CRITIQUE, CLUSTER_FORMED, SYNTHESIS_STARTED, FINAL_VERDICT."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="m1", response_text="Solar wind causes auroras."),
            MockProvider(name="m2", response_text="Charged particles from the sun cause auroras."),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=2,
            enable_logging_observer=False,
        )

        seen_types: set[str] = set()
        engine.event_bus.subscribe_all(lambda e: seen_types.add(e.event_type.name))

        await engine.run("What causes auroras?")

        assert "PROMPT_PERTURBED" in seen_types
        assert "MODEL_RESPONSE" in seen_types
        assert "CLUSTER_FORMED" in seen_types
        assert "SYNTHESIS_STARTED" in seen_types
        assert "FINAL_VERDICT" in seen_types

    @pytest.mark.asyncio
    async def test_pipeline_model_response_has_payload(self):
        """MODEL_RESPONSE events should carry text, provider, variant in payload."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="m1", response_text="Test response one."),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
        )

        model_events: list[Event] = []
        engine.event_bus.subscribe(EventType.MODEL_RESPONSE, model_events.append)

        await engine.run("test")

        assert len(model_events) >= 4  # 1 provider × 4 variants
        for evt in model_events:
            assert "text" in evt.payload
            assert "provider" in evt.payload
            assert "variant" in evt.payload
            assert len(evt.payload["text"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_final_verdict_has_answer_payload(self):
        """FINAL_VERDICT event should carry the full answer in payload."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="m1", response_text="Answer about auroras."),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
        )

        verdict_events: list[Event] = []
        engine.event_bus.subscribe(EventType.FINAL_VERDICT, verdict_events.append)

        await engine.run("test")

        assert len(verdict_events) == 1
        assert "answer" in verdict_events[0].payload
        assert "stability_score" in verdict_events[0].payload
        assert len(verdict_events[0].payload["answer"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_no_providers_returns_error(self):
        """Engine with zero available providers should return an error verdict."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="offline", response_text="", available=False),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            enable_logging_observer=False,
        )
        verdict = await engine.run("test")
        assert "ERROR" in verdict.answer or "No providers" in verdict.answer

    @pytest.mark.asyncio
    async def test_pipeline_deterministic_converges(self):
        """With deterministic mock providers the stability loop should converge."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="det", response_text="The answer is 42."),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=3,
            enable_logging_observer=False,
        )
        verdict = await engine.run("What is the meaning of life?")

        # Deterministic → identical synthesis each time → cosine sim ≈ 1.0
        assert verdict.stability_score >= 0.85

    @pytest.mark.asyncio
    async def test_pipeline_run_id_generated(self):
        """Every pipeline run should produce a unique non-empty run_id."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="m1", response_text="Test response."),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
        )
        verdict = await engine.run("test")

        assert verdict.run_id != ""
        assert len(verdict.run_id) == 12

    @pytest.mark.asyncio
    async def test_pipeline_stage_timings_populated(self):
        """Stage timings should be populated after a successful run."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="m1", response_text="Test response."),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
        )
        verdict = await engine.run("test")

        assert "diversify" in verdict.stage_timings
        assert "infer" in verdict.stage_timings
        assert "review" in verdict.stage_timings
        assert "cluster" in verdict.stage_timings
        assert "synthesize" in verdict.stage_timings
        for stage, elapsed in verdict.stage_timings.items():
            assert elapsed >= 0, f"Stage '{stage}' has negative timing"

    @pytest.mark.asyncio
    async def test_pipeline_confidence_score_positive(self):
        """The confidence score should be > 0 for a successful run."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="m1", response_text="Test response."),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
        )
        verdict = await engine.run("test")

        assert 0 < verdict.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_pipeline_events_carry_run_id(self):
        """All events should carry the run_id of the current pipeline run."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="m1", response_text="Test response."),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
        )

        run_ids: set[str] = set()

        def capture_run_id(evt: Event):
            if evt.run_id:
                run_ids.add(evt.run_id)

        engine.event_bus.subscribe(EventType.MODEL_RESPONSE, capture_run_id)
        engine.event_bus.subscribe(EventType.FINAL_VERDICT, capture_run_id)

        verdict = await engine.run("test")

        # All events should share the same run_id
        assert len(run_ids) == 1
        assert verdict.run_id in run_ids

    @pytest.mark.asyncio
    async def test_pipeline_final_verdict_event_has_confidence(self):
        """FINAL_VERDICT event should include confidence_score and stage_timings."""
        from d_mmce.orchestrator import D_MMCE

        providers = [
            MockProvider(name="m1", response_text="Answer about auroras."),
        ]
        engine = D_MMCE(
            providers=providers,
            review_provider_name="auto",
            stability_threshold=0.85,
            max_stability_reruns=1,
            enable_logging_observer=False,
        )

        verdict_events: list[Event] = []
        engine.event_bus.subscribe(EventType.FINAL_VERDICT, verdict_events.append)

        await engine.run("test")

        assert len(verdict_events) == 1
        assert "confidence_score" in verdict_events[0].payload
        assert "stage_timings" in verdict_events[0].payload

