"""
Ollama Provider – Local LLMs
==============================
Async wrapper around a locally-running Ollama server via its HTTP API.

Supports **any** model available in the local Ollama instance — the user
can pick from the list returned by ``OllamaProvider.list_local_models()``
or specify a model tag directly (e.g. ``"mistral"``, ``"codellama:13b"``,
``"phi3"``).

Each Ollama model registers as a distinct provider named ``ollama:<tag>``
so multiple local models can participate in the ensemble simultaneously.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from d_mmce.providers.base import ModelProvider
from d_mmce.providers.factory import register

logger = logging.getLogger(__name__)


def _ollama_base_url() -> str:
    """Resolve the Ollama base URL from env or default."""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@register("ollama")
class OllamaProvider(ModelProvider):
    """Strategy implementation for a local Ollama instance.

    Can target **any** locally-available model.  The provider ``name`` is
    set to ``ollama:<model>`` so the pipeline can distinguish between
    multiple local models running side by side.

    Parameters
    ----------
    base_url : str, optional
        Ollama HTTP endpoint (default from ``OLLAMA_BASE_URL`` env var or
        ``http://localhost:11434``).
    model : str
        Model tag to use (default ``"llama3.1"``).  Any tag that appears
        in ``ollama list`` is valid.
    timeout : float
        Request timeout in seconds (default ``300``).  Large models
        (e.g. 20B+) can take several minutes per generation.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "llama3.1",
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url or _ollama_base_url()
        self._model = model
        self._timeout = timeout
        # Dynamic name so the pipeline can tell models apart
        self.name = f"ollama:{model}"

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
        """Check if the Ollama server is reachable and the model exists."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                models = resp.json().get("models", [])
                available_tags = {m.get("name", "").split(":")[0] for m in models}
                # Also check full tag (e.g. "codellama:13b")
                available_full = {m.get("name", "") for m in models}
                model_base = self._model.split(":")[0]
                return (
                    self._model in available_full
                    or model_base in available_tags
                )
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ------------------------------------------------------------------ #
    #  Local model discovery
    # ------------------------------------------------------------------ #

    @staticmethod
    async def list_local_models(
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query the Ollama server for all locally-available models.

        Returns a list of dicts, each containing at least:
        - ``name``:             Full model tag (e.g. ``"llama3.1:latest"``)
        - ``short_name``:       Base name without tag suffix (e.g. ``"llama3.1"``)
        - ``size``:             Model size in bytes
        - ``parameter_size``:   Human-readable param count (e.g. ``"8B"``)
        - ``quantization``:     Quantization level (e.g. ``"Q4_0"``)
        - ``modified_at``:      Last modification timestamp

        Returns an empty list if Ollama is unreachable.
        """
        url = base_url or _ollama_base_url()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{url}/api/tags")
                if resp.status_code != 200:
                    return []
                data = resp.json()
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return []

        result: list[dict[str, Any]] = []
        for m in data.get("models", []):
            full_name = m.get("name", "")
            details = m.get("details", {})
            result.append({
                "name": full_name,
                "short_name": full_name.split(":")[0] if full_name else "",
                "size": m.get("size", 0),
                "parameter_size": details.get("parameter_size", ""),
                "quantization": details.get("quantization_level", ""),
                "family": details.get("family", ""),
                "modified_at": m.get("modified_at", ""),
            })
        return result

    @staticmethod
    async def server_reachable(base_url: str | None = None) -> bool:
        """Quick check whether any Ollama server is responding."""
        url = base_url or _ollama_base_url()
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

