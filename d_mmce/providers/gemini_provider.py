"""
Google Provider – Gemini 1.5
=============================
Async wrapper around the ``google-generativeai`` SDK.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import google.generativeai as genai

from d_mmce.providers.base import ModelProvider
from d_mmce.providers.factory import register


@register("gemini")
class GeminiProvider(ModelProvider):
    """Strategy implementation for Google Gemini 1.5 Pro.

    Parameters
    ----------
    api_key : str, optional
        Falls back to the ``GOOGLE_API_KEY`` environment variable.
    model : str
        Model identifier (default ``"gemini-1.5-pro"``).
    temperature : float
        Sampling temperature (default ``0.7``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7,
    ) -> None:
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY", ""))
        self._model_name = model
        self._temperature = temperature
        self._model = genai.GenerativeModel(model)

    async def _call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        # google-generativeai uses a sync API; run in a thread executor.
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self._temperature,
                ),
            ),
        )
        text = response.text or ""
        meta = {
            "model": self._model_name,
            "candidates": len(response.candidates) if response.candidates else 0,
        }
        return text, meta

