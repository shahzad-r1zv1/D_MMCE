"""
Unit tests for D-MMCE core components.
Run with:  pytest tests/test_unit.py -v
"""
from __future__ import annotations

import pytest

from d_mmce.observer import Event, EventBus, EventType
from d_mmce.prompt_perturbator import PromptPerturbator
from d_mmce.schemas import (
    ContradictionMatrix,
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

