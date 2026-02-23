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
import random
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

    In **single-reviewer mode** (default), a single designated model
    performs all critiques.  The ``reviewer`` field of each
    :class:`Critique` is set to the **actual** reviewing model's name,
    and ``reviewee`` indicates which provider's response was evaluated.

    Parameters
    ----------
    review_provider : ModelProvider
        The model used to perform the actual critique calls.  Using a
        single designated reviewer keeps cost predictable; alternatively
        pass the same pool to do round-robin reviews.
    event_bus : EventBus, optional
        Publish :pydata:`EventType.PEER_CRITIQUE` events.
    max_retries : int
        Number of retries for each critique call (default ``2``).
    retry_base_delay : float
        Base delay in seconds for jittered exponential backoff (default ``2.0``).
    request_timeout : float
        Per-request timeout in seconds (default ``120.0``).
    """

    def __init__(
        self,
        review_provider: ModelProvider,
        event_bus: EventBus | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 2.0,
        request_timeout: float = 120.0,
    ) -> None:
        self._reviewer = review_provider
        self._bus = event_bus
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._request_timeout = request_timeout

    async def _call_with_retry(self, prompt: str, variant: str) -> ModelResponse:
        """Call the review provider with timeout and jittered exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                coro = self._reviewer.generate(prompt, variant=variant)
                return await asyncio.wait_for(coro, timeout=self._request_timeout)
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    jitter = random.uniform(0, self._retry_base_delay)
                    delay = self._retry_base_delay * (2 ** attempt) + jitter
                    logger.warning(
                        "Peer-review retry %d/%d for %s — %s (delay %.1fs)",
                        attempt + 1, self._max_retries,
                        self._reviewer.name, type(exc).__name__, delay,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def review(
        self,
        responses: Sequence[ModelResponse],
        providers: Sequence[ModelProvider] | None = None,
    ) -> ContradictionMatrix:
        """Run peer reviews and return the contradiction matrix.

        In single-reviewer mode each unique provider's *best* response
        is reviewed once by the designated ``review_provider``.  The
        ``Critique.reviewer`` field is set to the **actual** model name,
        and ``Critique.critique_source`` is also populated.

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

        # Build critique tasks — one per reviewee (single-reviewer mode)
        tasks: list[tuple[str, asyncio.Task]] = []
        for reviewee_name in provider_names:
            prompt = PEER_REVIEW_PROMPT.format(
                solution=best_by_provider[reviewee_name].text
            )
            task = asyncio.create_task(
                self._call_with_retry(prompt, variant="peer_review")
            )
            tasks.append((reviewee_name, task))

        # Gather all critique results
        actual_reviewer = self._reviewer.name
        for reviewee_name, task in tasks:
            try:
                result: ModelResponse = await task
                is_validated, issues = _parse_critique(result.text)
                penalty = _penalty_from_issues(issues, is_validated)

                critique = Critique(
                    reviewer=actual_reviewer,
                    reviewee=reviewee_name,
                    critique_text=result.text,
                    is_validated=is_validated,
                    issues=issues,
                    critique_source=actual_reviewer,
                )
                matrix.critiques.append(critique)
                matrix.scores[(actual_reviewer, reviewee_name)] = penalty

                if self._bus:
                    self._bus.publish(
                        Event(
                            EventType.PEER_CRITIQUE,
                            message=(
                                f"{actual_reviewer} reviewed {reviewee_name}: "
                                f"{'VALIDATED' if is_validated else f'{len(issues)} issues'}"
                            ),
                            payload={
                                "reviewer": actual_reviewer,
                                "reviewee": reviewee_name,
                                "critique_source": actual_reviewer,
                                "penalty": penalty,
                                "issues": issues,
                            },
                        )
                    )
            except Exception:
                logger.warning(
                    "Critique %s→%s failed", actual_reviewer, reviewee_name,
                    exc_info=True,
                )
                matrix.scores[(actual_reviewer, reviewee_name)] = 0.5  # neutral

        return matrix

