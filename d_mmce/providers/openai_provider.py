"""
OpenAI Provider – GPT-4o
=========================
Async wrapper around the ``openai`` Python SDK.
"""

from __future__ import annotations

import os
from typing import Any

import openai

from d_mmce.providers.base import ModelProvider
from d_mmce.providers.factory import register


@register("openai")
class OpenAIProvider(ModelProvider):
    """Strategy implementation for OpenAI GPT-4o.

    Parameters
    ----------
    api_key : str, optional
        Falls back to the ``OPENAI_API_KEY`` environment variable.
    model : str
        Model identifier (default ``"gpt-4o"``).
    temperature : float
        Sampling temperature (default ``0.7``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = openai.AsyncOpenAI(api_key=self._api_key)
        self._model = model
        self._temperature = temperature

    async def is_available(self) -> bool:
        """Check if a valid API key is configured."""
        return bool(self._api_key and self._api_key.startswith("sk-"))

    async def _call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        choice = response.choices[0]
        meta = {
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage else {},
            "model": response.model,
        }
        return choice.message.content or "", meta

