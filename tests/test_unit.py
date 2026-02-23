"""
Unit tests for D-MMCE core components.
Run with:  pytest tests/test_unit.py -v
"""
from __future__ import annotations

import asyncio

import pytest

from d_mmce.observer import Event, EventBus, EventType
from d_mmce.prompt_perturbator import PromptPerturbator
from d_mmce.schemas import (
    ContradictionMatrix,
    FailureCategory,
    FinalVerdict,
    ModelResponse,
    PerturbedPrompt,
)

# =====================================================================
# Schema tests
# =====================================================================


class TestPerturbedPrompt:
    def test_variants_returns_four_keys(self):
        pp = PerturbedPrompt(
            original="Q", step_by_step="S", adversarial="A", summary="U"
        )
        v = pp.variants()
        assert set(v.keys()) == {"original", "step_by_step", "adversarial", "summary"}
        assert len(v) == 4

    def test_frozen(self):
        pp = PerturbedPrompt(original="Q", step_by_step="S", adversarial="A", summary="U")
        with pytest.raises(AttributeError):
            pp.original = "changed"  # type: ignore[misc]


class TestModelResponse:
    def test_uid(self):
        r = ModelResponse(provider_name="openai", prompt_variant="step_by_step", text="hi")
        assert r.uid == "openai::step_by_step"

    def test_default_latency(self):
        r = ModelResponse(provider_name="x", prompt_variant="y", text="z")
        assert r.latency == 0.0


class TestContradictionMatrix:
    def test_avg_penalty_with_scores(self):
        m = ContradictionMatrix()
        m.scores[("a", "b")] = 0.5
        m.scores[("c", "b")] = 0.3
        assert m.avg_penalty("b") == pytest.approx(0.4)

    def test_avg_penalty_empty(self):
        m = ContradictionMatrix()
        assert m.avg_penalty("unknown") == 0.0

    def test_avg_penalty_no_reviews_for_provider(self):
        m = ContradictionMatrix()
        m.scores[("a", "b")] = 0.8
        assert m.avg_penalty("a") == 0.0  # 'a' was reviewer, not reviewee


class TestFinalVerdict:
    def test_defaults(self):
        v = FinalVerdict(answer="Hello")
        assert v.stability_score == 1.0
        assert v.num_reruns == 0
        assert v.audit_trail == []
        assert v.run_id == ""
        assert v.stage_timings == {}
        assert v.confidence_score == 0.0

    def test_run_id_and_timings(self):
        v = FinalVerdict(
            answer="ok",
            run_id="abc123",
            stage_timings={"infer": 1.5, "review": 0.8},
            confidence_score=0.85,
        )
        assert v.run_id == "abc123"
        assert v.stage_timings["infer"] == 1.5
        assert v.confidence_score == 0.85


# =====================================================================
# EventBus tests
# =====================================================================


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        captured: list[Event] = []
        bus.subscribe(EventType.MODEL_RESPONSE, captured.append)

        evt = Event(EventType.MODEL_RESPONSE, message="test")
        bus.publish(evt)
        assert len(captured) == 1
        assert captured[0].message == "test"

    def test_subscribe_all(self):
        bus = EventBus()
        captured: list[Event] = []
        bus.subscribe_all(captured.append)

        for et in EventType:
            bus.publish(Event(et, message=et.name))

        assert len(captured) == len(EventType)

    def test_publish_wrong_type_not_received(self):
        bus = EventBus()
        captured: list[Event] = []
        bus.subscribe(EventType.FINAL_VERDICT, captured.append)

        bus.publish(Event(EventType.MODEL_RESPONSE, message="miss"))
        assert len(captured) == 0

    def test_subscriber_exception_does_not_break_bus(self):
        bus = EventBus()
        captured: list[str] = []

        def bad(evt: Event):
            raise RuntimeError("boom")

        def good(evt: Event):
            captured.append(evt.message)

        bus.subscribe(EventType.CLUSTER_FORMED, bad)
        bus.subscribe(EventType.CLUSTER_FORMED, good)

        bus.publish(Event(EventType.CLUSTER_FORMED, message="ok"))
        assert captured == ["ok"]  # good still runs


# =====================================================================
# PromptPerturbator tests
# =====================================================================


class TestPromptPerturbator:
    def test_perturb_returns_four_variants(self):
        pp = PromptPerturbator()
        result = pp.perturb("What is gravity?")
        assert isinstance(result, PerturbedPrompt)
        v = result.variants()
        assert len(v) == 4
        assert result.original == "What is gravity?"

    def test_variants_contain_query(self):
        pp = PromptPerturbator()
        result = pp.perturb("Why is the sky blue?")
        for name, text in result.variants().items():
            assert "Why is the sky blue?" in text, f"Variant '{name}' missing query"

    def test_emits_event(self, event_bus):
        captured: list[Event] = []
        event_bus.subscribe(EventType.PROMPT_PERTURBED, captured.append)

        pp = PromptPerturbator(event_bus=event_bus)
        pp.perturb("test query")

        assert len(captured) == 1
        assert "test query" in captured[0].message


# =====================================================================
# ModelProvider (base) tests
# =====================================================================


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_generate_returns_model_response(self, mock_provider):
        resp = await mock_provider.generate("hello", variant="original")
        assert isinstance(resp, ModelResponse)
        assert resp.provider_name == "mock-alpha"
        assert resp.prompt_variant == "original"
        assert resp.text == "The Northern Lights are caused by solar particles."
        assert resp.latency > 0
        assert mock_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_is_available(self, mock_provider, mock_unavailable):
        assert await mock_provider.is_available() is True
        assert await mock_unavailable.is_available() is False


# =====================================================================
# PeerReviewer unit tests
# =====================================================================


class TestPeerReviewerParsing:
    """Test the internal _parse_critique helper."""

    def test_validated(self):
        from d_mmce.peer_reviewer import _parse_critique

        ok, issues = _parse_critique("VALIDATED")
        assert ok is True
        assert issues == []

    def test_validated_case_insensitive(self):
        from d_mmce.peer_reviewer import _parse_critique

        ok, _ = _parse_critique("Validated")
        assert ok is True

    def test_bullet_points(self):
        from d_mmce.peer_reviewer import _parse_critique

        text = "- First issue\n- Second issue\n- Third issue"
        ok, issues = _parse_critique(text)
        assert ok is False
        assert len(issues) == 3

    def test_numbered_points(self):
        from d_mmce.peer_reviewer import _parse_critique

        text = "1. Error one\n2. Error two"
        ok, issues = _parse_critique(text)
        assert ok is False
        assert len(issues) == 2

    def test_caps_at_five(self):
        from d_mmce.peer_reviewer import _parse_critique

        text = "\n".join(f"- Issue {i}" for i in range(10))
        ok, issues = _parse_critique(text)
        assert ok is False
        assert len(issues) <= 5


class TestPenaltyCalculation:
    def test_validated_zero_penalty(self):
        from d_mmce.peer_reviewer import _penalty_from_issues

        assert _penalty_from_issues([], True) == 0.0

    def test_one_issue(self):
        from d_mmce.peer_reviewer import _penalty_from_issues

        assert _penalty_from_issues(["a"], False) == 0.25

    def test_four_issues_capped(self):
        from d_mmce.peer_reviewer import _penalty_from_issues

        assert _penalty_from_issues(["a", "b", "c", "d"], False) == 1.0

    def test_five_issues_still_capped(self):
        from d_mmce.peer_reviewer import _penalty_from_issues

        assert _penalty_from_issues(["a"] * 5, False) == 1.0


# =====================================================================
# ProviderFactory tests
# =====================================================================


class TestProviderFactory:
    def test_available_names_includes_core(self):
        from d_mmce.providers.factory import ProviderFactory

        names = ProviderFactory.available_names()
        assert "openai" in names
        assert "anthropic" in names
        assert "gemini" in names
        assert "ollama" in names

    def test_create_unknown_raises(self):
        from d_mmce.providers.factory import ProviderFactory

        with pytest.raises(KeyError, match="Unknown provider"):
            ProviderFactory.create("totally_fake_provider")

    def test_create_ollama_shorthand(self):
        from d_mmce.providers.factory import ProviderFactory

        p = ProviderFactory.create("ollama:test-model")
        assert p.name == "ollama:test-model"


# =====================================================================
# FailureCategory tests
# =====================================================================


class TestFailureCategory:
    def test_enum_members(self):
        assert FailureCategory.TIMEOUT.name == "TIMEOUT"
        assert FailureCategory.AUTH.name == "AUTH"
        assert FailureCategory.QUOTA.name == "QUOTA"
        assert FailureCategory.TRANSIENT.name == "TRANSIENT"
        assert FailureCategory.PERMANENT.name == "PERMANENT"


# =====================================================================
# classify_failure tests
# =====================================================================


class TestClassifyFailure:
    def test_timeout_error(self):
        from d_mmce.orchestrator import classify_failure

        assert classify_failure(asyncio.TimeoutError()) == FailureCategory.TIMEOUT

    def test_timeout_in_message(self):
        from d_mmce.orchestrator import classify_failure

        assert classify_failure(Exception("Request timeout")) == FailureCategory.TIMEOUT

    def test_auth_error(self):
        from d_mmce.orchestrator import classify_failure

        assert classify_failure(Exception("401 Unauthorized")) == FailureCategory.AUTH

    def test_auth_api_key(self):
        from d_mmce.orchestrator import classify_failure

        assert classify_failure(Exception("Invalid API key")) == FailureCategory.AUTH

    def test_quota_429(self):
        from d_mmce.orchestrator import classify_failure

        assert classify_failure(Exception("429 rate limit exceeded")) == FailureCategory.QUOTA

    def test_transient_generic(self):
        from d_mmce.orchestrator import classify_failure

        assert classify_failure(ConnectionError("connection reset")) == FailureCategory.TRANSIENT

    def test_permanent_400(self):
        from d_mmce.orchestrator import classify_failure

        assert classify_failure(Exception("400 bad request")) == FailureCategory.PERMANENT

    def test_is_retryable(self):
        from d_mmce.orchestrator import _is_retryable

        assert _is_retryable(FailureCategory.TIMEOUT) is True
        assert _is_retryable(FailureCategory.QUOTA) is True
        assert _is_retryable(FailureCategory.TRANSIENT) is True
        assert _is_retryable(FailureCategory.AUTH) is False
        assert _is_retryable(FailureCategory.PERMANENT) is False


# =====================================================================
# Critique schema tests
# =====================================================================


class TestCritique:
    def test_critique_source_default(self):
        from d_mmce.schemas import Critique

        c = Critique(reviewer="r", reviewee="e", critique_text="text")
        assert c.critique_source == ""

    def test_critique_source_set(self):
        from d_mmce.schemas import Critique

        c = Critique(
            reviewer="actual-model",
            reviewee="target",
            critique_text="ok",
            critique_source="actual-model",
        )
        assert c.critique_source == "actual-model"


# =====================================================================
# ConsensusCluster schema tests
# =====================================================================


class TestConsensusCluster:
    def test_insufficient_consensus_default(self):
        from d_mmce.schemas import ConsensusCluster

        cc = ConsensusCluster(centroid_text="test")
        assert cc.insufficient_consensus is False
        assert cc.consensus_ratio == 1.0

    def test_insufficient_consensus_flag(self):
        from d_mmce.schemas import ConsensusCluster

        cc = ConsensusCluster(
            centroid_text="test",
            insufficient_consensus=True,
            consensus_ratio=0.2,
        )
        assert cc.insufficient_consensus is True
        assert cc.consensus_ratio == 0.2


# =====================================================================
# Event run_id tests
# =====================================================================


class TestEventRunId:
    def test_event_run_id_default(self):
        evt = Event(EventType.MODEL_RESPONSE, message="test")
        assert evt.run_id == ""

    def test_event_run_id_set(self):
        evt = Event(EventType.MODEL_RESPONSE, message="test", run_id="abc123")
        assert evt.run_id == "abc123"


# =====================================================================
# Peer reviewer parsing edge cases
# =====================================================================


class TestPeerReviewerParsingEdgeCases:
    def test_empty_string(self):
        from d_mmce.peer_reviewer import _parse_critique

        ok, issues = _parse_critique("")
        assert ok is False
        assert issues == []

    def test_whitespace_only(self):
        from d_mmce.peer_reviewer import _parse_critique

        ok, issues = _parse_critique("   \n  \n  ")
        assert ok is False
        assert issues == []

    def test_validated_with_trailing_text(self):
        from d_mmce.peer_reviewer import _parse_critique

        ok, issues = _parse_critique("VALIDATED - looks good")
        assert ok is True

    def test_mixed_bullet_styles(self):
        from d_mmce.peer_reviewer import _parse_critique

        text = "- First\n* Second\n• Third\n1. Fourth"
        ok, issues = _parse_critique(text)
        assert ok is False
        assert len(issues) == 4

    def test_asterisk_bullets(self):
        from d_mmce.peer_reviewer import _parse_critique

        text = "* Issue A\n* Issue B"
        ok, issues = _parse_critique(text)
        assert ok is False
        assert len(issues) == 2

    def test_parenthetical_numbering(self):
        from d_mmce.peer_reviewer import _parse_critique

        text = "1) Error one\n2) Error two"
        ok, issues = _parse_critique(text)
        assert ok is False
        assert len(issues) == 2


# =====================================================================
# Penalty scoring edge cases
# =====================================================================


class TestPenaltyEdgeCases:
    def test_zero_issues_not_validated(self):
        from d_mmce.peer_reviewer import _penalty_from_issues

        assert _penalty_from_issues([], False) == 0.0

    def test_two_issues(self):
        from d_mmce.peer_reviewer import _penalty_from_issues

        assert _penalty_from_issues(["a", "b"], False) == 0.5

    def test_three_issues(self):
        from d_mmce.peer_reviewer import _penalty_from_issues

        assert _penalty_from_issues(["a", "b", "c"], False) == 0.75

