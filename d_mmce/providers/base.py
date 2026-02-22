"""
Model Provider – Strategy Interface (Abstract Base)
====================================================
Every LLM backend implements this interface so the orchestrator can treat
them interchangeably (Strategy Pattern).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from d_mmce.schemas import ModelResponse


class ModelProvider(ABC):
    """Abstract async LLM provider.

    Subclasses must implement :meth:`_call` which performs the raw API
    request and returns the generated text.  The public :meth:`generate`
    method wraps ``_call`` with latency measurement and uniform
    :class:`ModelResponse` construction.

    Attributes
    ----------
    name : str
        Canonical provider name used as a key throughout the pipeline.
    """

    name: str = "base"

    @abstractmethod
    async def _call(self, prompt: str) -> tuple[str, dict]:
        """Perform the actual API call.

        Returns
        -------
        tuple[str, dict]
            ``(generated_text, metadata_dict)``
        """
        ...

    async def generate(self, prompt: str, variant: str = "original") -> ModelResponse:
        """Generate a response, measuring latency.

        Parameters
        ----------
        prompt : str
            The fully-formed prompt text.
        variant : str
            Label for the prompt variant (``"original"``, ``"step_by_step"``, …).
        """
        t0 = time.perf_counter()
        text, meta = await self._call(prompt)
        elapsed = time.perf_counter() - t0
        return ModelResponse(
            provider_name=self.name,
            prompt_variant=variant,
            text=text,
            latency=elapsed,
            metadata=meta,
        )

    async def is_available(self) -> bool:
        """Quick health-check – override for providers that may be offline."""
        return True

