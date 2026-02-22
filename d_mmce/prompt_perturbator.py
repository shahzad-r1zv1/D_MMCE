"""
Prompt Perturbator – Diversity Injection
=========================================
Takes a single user query and produces three semantically distinct variations,
each designed to elicit a different *reasoning style* from the model pool:

* **Step-by-step** – forces chain-of-thought decomposition.
* **Adversarial** – challenges hidden assumptions and edge cases.
* **Summary** – demands a direct, concise answer.

The original query is also preserved so we always have four prompt variants.
"""

from __future__ import annotations

from d_mmce.observer import Event, EventBus, EventType
from d_mmce.schemas import PerturbedPrompt

# ──────────────────────────────────────────────────────────────────────
# Prompt templates  –  {query} is replaced at runtime
# ──────────────────────────────────────────────────────────────────────

STEP_BY_STEP_TEMPLATE = (
    "Please answer the following question by breaking it down into clear, "
    "numbered steps. Show your reasoning at every stage.\n\n"
    "Question: {query}"
)

ADVERSARIAL_TEMPLATE = (
    "You are a critical thinker. Challenge any hidden assumptions in the "
    "following question and consider edge cases or counter-arguments before "
    "providing your answer.\n\n"
    "Question: {query}"
)

SUMMARY_TEMPLATE = (
    "Provide a concise, direct answer to the following question in no more "
    "than three sentences.\n\n"
    "Question: {query}"
)


class PromptPerturbator:
    """Generate semantically diverse prompt variants from a single query.

    This is a **pure-function** component – no LLM call required.  Templates
    are deterministic so the diversity comes from how each downstream model
    *interprets* a different framing, not from randomness here.

    Parameters
    ----------
    event_bus : EventBus, optional
        If provided, an :pydata:`EventType.PROMPT_PERTURBED` event is
        published after each perturbation.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._bus = event_bus

    def perturb(self, query: str) -> PerturbedPrompt:
        """Return a :class:`PerturbedPrompt` with all four variants."""
        prompt = PerturbedPrompt(
            original=query,
            step_by_step=STEP_BY_STEP_TEMPLATE.format(query=query),
            adversarial=ADVERSARIAL_TEMPLATE.format(query=query),
            summary=SUMMARY_TEMPLATE.format(query=query),
        )
        if self._bus:
            self._bus.publish(
                Event(
                    EventType.PROMPT_PERTURBED,
                    message=f"Generated 3 variants for query: {query[:80]}…",
                    payload={"variants": list(prompt.variants().keys())},
                )
            )
        return prompt

