"""
Meta-Judge – Global Synthesis & Stability Loop
================================================
The final pipeline stage.  The **Meta-Judge** receives:

* The **Consensus Cluster** (semantically-agreed answers).
* The **Contradiction Matrix** (peer-review penalties).

It selects the *highest-weighted* provider (lowest average penalty) as the
synthesis model, prompts it to produce a unified answer, and then runs a
**Stability Loop** to ensure the answer doesn't contradict itself across
successive iterations.

Stability Loop — Global vs. Local Optimality
---------------------------------------------
A *Local Optimum* is an answer that looks correct in isolation but diverges
from the ensemble consensus.  A *Global Optimum* is the answer that
remains **stable** when re-verified against the full evidence.

The loop works as follows:

1. Generate a candidate synthesis from the consensus cluster.
2. Compute the cosine similarity between the new synthesis and the
   previous one (if any).
3. If similarity ≥ ``stability_threshold`` → **converged** (Global Optimum).
4. Otherwise, re-run the synthesis (up to ``max_reruns``).

If the loop exhausts its budget, the last candidate is returned with a low
``stability_score`` to signal that the system did not converge — the caller
can choose to flag this answer as uncertain.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from d_mmce.observer import Event, EventBus, EventType
from d_mmce.providers.base import ModelProvider
from d_mmce.schemas import (
    ConsensusCluster,
    ContradictionMatrix,
    FinalVerdict,
    ModelResponse,
)

if TYPE_CHECKING:
    from d_mmce.semantic_clusterer import SemanticClusterer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Synthesis prompt
# ──────────────────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = (
    "You are a Meta-Judge tasked with producing the single best answer.\n\n"
    "Below are multiple AI-generated answers that have been verified by "
    "peer review and clustered by semantic similarity. They represent the "
    "ensemble's consensus.\n\n"
    "--- CONSENSUS RESPONSES ---\n"
    "{consensus_block}\n"
    "--- END CONSENSUS RESPONSES ---\n\n"
    "--- PEER REVIEW SUMMARY ---\n"
    "{review_summary}\n"
    "--- END PEER REVIEW SUMMARY ---\n\n"
    "Synthesise these into a single, coherent, and accurate answer. "
    "Resolve any minor differences. If there are contradictions, side with "
    "the majority. Provide only the final answer."
)


def _build_consensus_block(cluster: ConsensusCluster) -> str:
    """Format member responses into a numbered block."""
    lines: list[str] = []
    for i, r in enumerate(cluster.member_responses, 1):
        lines.append(f"[{i}] ({r.provider_name}) {r.text}")
    return "\n\n".join(lines) if lines else "(no consensus responses)"


def _build_review_summary(matrix: ContradictionMatrix) -> str:
    """One-line summary per provider from the contradiction matrix."""
    if not matrix.critiques:
        return "No peer reviews available."

    lines: list[str] = []
    providers = {c.reviewee for c in matrix.critiques}
    for p in sorted(providers):
        avg = matrix.avg_penalty(p)
        status = "VALIDATED" if avg == 0 else f"avg penalty {avg:.2f}"
        lines.append(f"- {p}: {status}")
    return "\n".join(lines)


class MetaJudge:
    """Final synthesis model with a Stability Loop.

    Parameters
    ----------
    providers : Sequence[ModelProvider]
        Full provider pool — the one with the lowest contradiction penalty
        will be used as the synthesis model.
    clusterer : SemanticClusterer
        A :class:`SemanticClusterer` instance (used for stability cosine
        similarity checks).
    stability_threshold : float
        Minimum cosine similarity between successive syntheses to declare
        convergence (default ``0.85``).
    max_reruns : int
        Maximum number of re-synthesis iterations (default ``3``).
    event_bus : EventBus, optional
        Publish synthesis and stability events.
    """

    def __init__(
        self,
        providers: Sequence[ModelProvider],
        clusterer: SemanticClusterer,
        stability_threshold: float = 0.85,
        max_reruns: int = 3,
        event_bus: EventBus | None = None,
    ) -> None:
        self._providers = {p.name: p for p in providers}
        self._clusterer = clusterer
        self._stability_threshold = stability_threshold
        self._max_reruns = max_reruns
        self._bus = event_bus

    def _select_judge(self, matrix: ContradictionMatrix) -> ModelProvider:
        """Pick the provider with the lowest average penalty."""
        best_name: str | None = None
        best_penalty = float("inf")

        for name in self._providers:
            penalty = matrix.avg_penalty(name)
            if penalty < best_penalty:
                best_penalty = penalty
                best_name = name

        if best_name is None:
            # Fallback: pick the first provider
            best_name = next(iter(self._providers))

        logger.info("Meta-Judge selected: %s (penalty %.2f)", best_name, best_penalty)
        return self._providers[best_name]

    async def synthesize(
        self,
        cluster: ConsensusCluster,
        matrix: ContradictionMatrix,
    ) -> FinalVerdict:
        """Produce the globally optimal answer via synthesis + stability loop.

        Returns
        -------
        FinalVerdict
            Contains the final answer, stability score, re-run count, and
            an audit trail.
        """
        judge = self._select_judge(matrix)
        prompt = SYNTHESIS_PROMPT.format(
            consensus_block=_build_consensus_block(cluster),
            review_summary=_build_review_summary(matrix),
        )

        audit: list[str] = []
        previous_text: str | None = None
        stability_score = 0.0

        if self._bus:
            self._bus.publish(
                Event(
                    EventType.SYNTHESIS_STARTED,
                    message=f"Meta-Judge ({judge.name}) beginning synthesis.",
                )
            )

        for attempt in range(1, self._max_reruns + 1):
            response: ModelResponse = await judge.generate(prompt, variant="synthesis")
            current_text = response.text

            audit.append(
                f"[Attempt {attempt}] Judge={judge.name}, "
                f"length={len(current_text)} chars"
            )

            if previous_text is not None:
                similarity = self._clusterer.cosine_similarity_texts(
                    previous_text, current_text
                )
                stability_score = similarity
                audit.append(
                    f"  Stability check: cosine_sim={similarity:.4f} "
                    f"(threshold={self._stability_threshold})"
                )

                if self._bus:
                    self._bus.publish(
                        Event(
                            EventType.STABILITY_CHECK,
                            message=(
                                f"Attempt {attempt}: similarity={similarity:.4f}"
                            ),
                            payload={
                                "attempt": attempt,
                                "similarity": similarity,
                                "converged": similarity >= self._stability_threshold,
                            },
                        )
                    )

                if similarity >= self._stability_threshold:
                    audit.append("  → Converged (Global Optimum reached).")
                    return FinalVerdict(
                        answer=current_text,
                        stability_score=stability_score,
                        num_reruns=attempt - 1,
                        audit_trail=audit,
                    )
                else:
                    audit.append(
                        "  → Unstable (Local Optimum detected). Re-running…"
                    )
                    if self._bus:
                        self._bus.publish(
                            Event(
                                EventType.STABILITY_RERUN,
                                message=f"Re-running synthesis (attempt {attempt + 1}).",
                            )
                        )
            else:
                # First attempt — need at least two for a stability check
                stability_score = 1.0

            previous_text = current_text

        # Exhausted re-runs — return last result with low stability
        audit.append(
            f"Stability loop exhausted after {self._max_reruns} attempts. "
            f"Returning last synthesis (potential Local Optimum)."
        )
        return FinalVerdict(
            answer=previous_text or "",
            stability_score=stability_score,
            num_reruns=self._max_reruns,
            audit_trail=audit,
        )

