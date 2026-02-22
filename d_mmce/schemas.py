"""
D-MMCE Data Schemas
===================
Typed dataclasses that carry data between every pipeline stage.
All inter-stage communication uses these immutable structures so that each
component remains loosely coupled and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PerturbedPrompt:
    """Three semantic variations of the original user query.

    Attributes:
        original:       The raw query as submitted by the user.
        step_by_step:   A variation that asks for a step-by-step breakdown.
        adversarial:    A variation that asks the model to challenge assumptions.
        summary:        A variation that asks for a concise summary answer.
    """

    original: str
    step_by_step: str
    adversarial: str
    summary: str

    def variants(self) -> dict[str, str]:
        """Return a mapping of variant name → prompt text."""
        return {
            "original": self.original,
            "step_by_step": self.step_by_step,
            "adversarial": self.adversarial,
            "summary": self.summary,
        }


@dataclass
class ModelResponse:
    """Single response from one model for one prompt variant.

    Attributes:
        provider_name:  Canonical name of the provider (e.g. ``"openai"``).
        prompt_variant: Which variant was used (``"original"``, ``"step_by_step"``, …).
        text:           The generated text.
        latency:        Wall-clock seconds for the API call.
        metadata:       Any provider-specific metadata (token counts, finish reason, …).
    """

    provider_name: str
    prompt_variant: str
    text: str
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        """Unique identifier combining provider and variant."""
        return f"{self.provider_name}::{self.prompt_variant}"


@dataclass
class Critique:
    """A single peer-review critique of one model response by another model.

    Attributes:
        reviewer:       Provider name of the reviewing model.
        reviewee:       Provider name of the model being reviewed.
        critique_text:  Raw critique text produced by the reviewer.
        is_validated:   ``True`` if the reviewer found no issues (responded VALIDATED).
        issues:         List of identified issues / failure points.
    """

    reviewer: str
    reviewee: str
    critique_text: str
    is_validated: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class ContradictionMatrix:
    """Pairwise contradiction scores and the raw critiques behind them.

    The *scores* dict maps ``(reviewer, reviewee)`` → penalty ∈ [0, 1].
    A score of 0 means "fully validated"; 1 means "severe contradictions found".

    Attributes:
        scores:    Pairwise penalty scores.
        critiques: Full list of :class:`Critique` objects.
    """

    scores: dict[tuple[str, str], float] = field(default_factory=dict)
    critiques: list[Critique] = field(default_factory=list)

    def avg_penalty(self, provider_name: str) -> float:
        """Average penalty score *received* by ``provider_name``."""
        relevant = [v for (_, r), v in self.scores.items() if r == provider_name]
        return sum(relevant) / len(relevant) if relevant else 0.0


@dataclass
class ConsensusCluster:
    """Result of the semantic clustering step.

    Attributes:
        centroid_text:     The response text closest to the cluster centroid.
        member_responses:  Responses that belong to the consensus cluster.
        outliers:          Responses that were discarded as local optima.
        centroid_embedding: The embedding vector of the centroid (used for stability checks).
    """

    centroid_text: str
    member_responses: list[ModelResponse] = field(default_factory=list)
    outliers: list[ModelResponse] = field(default_factory=list)
    centroid_embedding: list[float] | None = None


@dataclass
class FinalVerdict:
    """The globally optimal answer produced by the Meta-Judge.

    The **Stability Loop** may trigger multiple synthesis rounds.  Each round
    is recorded so the caller can inspect the system's convergence behaviour.

    Attributes:
        answer:           The synthesised final answer.
        stability_score:  Cosine similarity between the last two synthesis
                          rounds (1.0 = perfectly stable).
        num_reruns:       How many synthesis re-runs were required.
        audit_trail:      Human-readable log of each pipeline stage.
    """

    answer: str
    stability_score: float = 1.0
    num_reruns: int = 0
    audit_trail: list[str] = field(default_factory=list)

