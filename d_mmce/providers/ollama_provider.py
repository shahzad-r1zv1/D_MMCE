"""
Ollama Provider – Local Llama 3.1
==================================
Async wrapper around a locally-running Ollama server via its HTTP API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from d_mmce.providers.base import ModelProvider
from d_mmce.providers.factory import register


@register("ollama")
class OllamaProvider(ModelProvider):
    """Strategy implementation for a local Ollama instance (Llama 3.1).

    Parameters
    ----------
    base_url : str
        Ollama HTTP endpoint (default ``"http://localhost:11434"``).
    model : str
        Model tag to pull/use (default ``"llama3.1"``).
    timeout : float
        Request timeout in seconds (default ``120``).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "llama3.1",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self._model = model
        self._timeout = timeout

    async def _call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/api/generate", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

        text = data.get("response", "")
        meta = {
            "model": data.get("model", self._model),
            "total_duration": data.get("total_duration"),
            "eval_count": data.get("eval_count"),
        }
        return text, meta

    async def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

