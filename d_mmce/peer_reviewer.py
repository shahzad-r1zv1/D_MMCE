"""
Peer Reviewer – Cross-Examination Layer (The Immune System)
============================================================
Implements a pairwise peer-review protocol where each model critiques
another model's output.  The result is a :class:`ContradictionMatrix` that
quantifies how many verified hallucinations or logical failures each
model exhibited.

The **Immune System** analogy: just as the adaptive immune system uses
cross-reactive T-cells to detect foreign agents, the Peer Reviewer uses
cross-model critiques to detect outputs that are *semantically foreign* to
the ensemble's consensus.

Observer Integration
--------------------
Every critique is published via the :class:`EventBus` so external loggers
or dashboards can subscribe without coupling to this module.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Sequence

from d_mmce.observer import Event, EventBus, EventType
from d_mmce.providers.base import ModelProvider
from d_mmce.schemas import ContradictionMatrix, Critique, ModelResponse

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Peer-Review Prompt
# ──────────────────────────────────────────────────────────────────────

PEER_REVIEW_PROMPT = (
    "You are a logic auditor. Below is a solution provided by another AI. "
    "Identify three potential points of failure or factual inaccuracies. "
    "If the solution is perfect, respond with 'VALIDATED'.\n\n"
    "--- BEGIN SOLUTION ---\n"
    "{solution}\n"
    "--- END SOLUTION ---\n\n"
    "Your critique (bullet points or 'VALIDATED'):"
)


def _parse_critique(raw: str) -> tuple[bool, list[str]]:
    """Parse a critique response into (is_validated, issues)."""
    stripped = raw.strip()
    if stripped.upper().startswith("VALIDATED") or stripped.upper() == "VALIDATED":
        return True, []

    # Extract bullet points or numbered items
    issues: list[str] = []
    for line in stripped.splitlines():
        line = line.strip()
        # Match lines starting with -, *, •, or a number followed by . or )
        if re.match(r"^[-*•]|\d+[.)]", line):
            issues.append(re.sub(r"^[-*•\d.)\s]+", "", line).strip())
        elif line and not line.startswith("VALIDATED"):
            issues.append(line)
    return False, issues[:5]  # cap at 5 issues


def _penalty_from_issues(issues: list[str], is_validated: bool) -> float:
    """Convert issue count to a penalty score in [0, 1]."""
    if is_validated:
        return 0.0
    # Each issue contributes 0.25, capped at 1.0
    return min(len(issues) * 0.25, 1.0)


class PeerReviewer:
    """Cross-examination module that generates a :class:`ContradictionMatrix`.

    For every unique pair of providers, the reviewer asks provider A to
    critique provider B's *best* response (the one for the ``"original"``
    variant, or the first available).

    Parameters
    ----------
    review_provider : ModelProvider
        The model used to perform the actual critique calls.  Using a
        single designated reviewer keeps cost predictable; alternatively
        pass the same pool to do round-robin reviews.
    event_bus : EventBus, optional
        Publish :pydata:`EventType.PEER_CRITIQUE` events.
    """

    def __init__(
        self,
        review_provider: ModelProvider,
        event_bus: EventBus | None = None,
    ) -> None:
        self._reviewer = review_provider
        self._bus = event_bus

    async def review(
        self,
        responses: Sequence[ModelResponse],
        providers: Sequence[ModelProvider] | None = None,
    ) -> ContradictionMatrix:
        """Run pairwise peer reviews and return the contradiction matrix.

        Parameters
        ----------
        responses : Sequence[ModelResponse]
            All model responses from the parallel-inference stage.
        providers : Sequence[ModelProvider], optional
            Not used directly for calling – the ``review_provider`` does all
            critique calls.  Included for future round-robin extension.

        Returns
        -------
        ContradictionMatrix
        """
        # Group responses by provider, keep only the "original" variant
        best_by_provider: dict[str, ModelResponse] = {}
        for r in responses:
            if r.provider_name not in best_by_provider or r.prompt_variant == "original":
                best_by_provider[r.provider_name] = r

        provider_names = list(best_by_provider.keys())
        matrix = ContradictionMatrix()

        # Build critique tasks for all pairs
        tasks: list[tuple[str, str, asyncio.Task]] = []
        for i, reviewer_name in enumerate(provider_names):
            for j, reviewee_name in enumerate(provider_names):
                if i == j:
                    continue
                prompt = PEER_REVIEW_PROMPT.format(
                    solution=best_by_provider[reviewee_name].text
                )
                task = asyncio.create_task(
                    self._reviewer.generate(prompt, variant="peer_review")
                )
                tasks.append((reviewer_name, reviewee_name, task))

        # Gather all critique results
        for reviewer_name, reviewee_name, task in tasks:
            try:
                result: ModelResponse = await task
                is_validated, issues = _parse_critique(result.text)
                penalty = _penalty_from_issues(issues, is_validated)

                critique = Critique(
                    reviewer=reviewer_name,
                    reviewee=reviewee_name,
                    critique_text=result.text,
                    is_validated=is_validated,
                    issues=issues,
                )
                matrix.critiques.append(critique)
                matrix.scores[(reviewer_name, reviewee_name)] = penalty

                if self._bus:
                    self._bus.publish(
                        Event(
                            EventType.PEER_CRITIQUE,
                            message=(
                                f"{reviewer_name} → {reviewee_name}: "
                                f"{'VALIDATED' if is_validated else f'{len(issues)} issues'}"
                            ),
                            payload={
                                "reviewer": reviewer_name,
                                "reviewee": reviewee_name,
                                "penalty": penalty,
                                "issues": issues,
                            },
                        )
                    )
            except Exception:
                logger.warning(
                    "Critique %s→%s failed", reviewer_name, reviewee_name,
                    exc_info=True,
                )
                matrix.scores[(reviewer_name, reviewee_name)] = 0.5  # neutral

        return matrix

