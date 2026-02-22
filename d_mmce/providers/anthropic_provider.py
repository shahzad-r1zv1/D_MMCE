"""
Anthropic Provider – Claude 3.5 Sonnet
========================================
Async wrapper around the ``anthropic`` Python SDK.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from d_mmce.providers.base import ModelProvider
from d_mmce.providers.factory import register


@register("anthropic")
class AnthropicProvider(ModelProvider):
    """Strategy implementation for Anthropic Claude 3.5 Sonnet.

    Parameters
    ----------
    api_key : str, optional
        Falls back to the ``ANTHROPIC_API_KEY`` environment variable.
    model : str
        Model identifier (default ``"claude-3-5-sonnet-20241022"``).
    temperature : float
        Sampling temperature (default ``0.7``).
    max_tokens : int
        Maximum tokens to generate (default ``4096``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = anthropic.AsyncAnthropic(
            api_key=self._api_key if self._api_key else None,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def is_available(self) -> bool:
        """Check if a valid API key is configured."""
        return bool(self._api_key and len(self._api_key) > 10)

    async def _call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        meta = {
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "model": response.model,
        }
        return text, meta

