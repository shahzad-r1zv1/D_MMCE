"""
Shared fixtures for D-MMCE test suite.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from d_mmce.observer import EventBus
from d_mmce.providers.base import ModelProvider
from d_mmce.schemas import ModelResponse


# ---------------------------------------------------------------------------
# Mock provider – returns canned responses, no network calls
# ---------------------------------------------------------------------------

class MockProvider(ModelProvider):
    """A deterministic provider for testing. Returns pre-set text."""

    def __init__(
        self,
        name: str = "mock",
        response_text: str = "Mock response.",
        available: bool = True,
        latency: float = 0.01,
    ) -> None:
        self.name = name
        self._response_text = response_text
        self._available = available
        self._latency = latency
        self.call_count = 0

    async def _call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        self.call_count += 1
        await asyncio.sleep(self._latency)
        return self._response_text, {"mock": True}

    async def is_available(self) -> bool:
        return self._available


class MockStreamingProvider(ModelProvider):
    """Mock provider that supports streaming (token-by-token)."""

    def __init__(self, name: str = "mock-stream", response_text: str = "Hello world") -> None:
        self.name = name
        self._response_text = response_text
        self.streamed_tokens: list[str] = []

    async def _call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        return self._response_text, {}

    async def _call_stream(self, prompt: str, on_token=None) -> tuple[str, dict[str, Any]]:
        tokens = list(self._response_text)  # character-by-character
        self.streamed_tokens = []
        for tok in tokens:
            self.streamed_tokens.append(tok)
            if on_token:
                on_token(tok)
            await asyncio.sleep(0.001)
        return self._response_text, {"streamed": True}

    async def is_available(self) -> bool:
        return True


class FlakyProvider(ModelProvider):
    """Fails N times then succeeds. For testing retry logic."""

    def __init__(self, name: str = "flaky", response_text: str = "Success!", fail_count: int = 2) -> None:
        self.name = name
        self._response_text = response_text
        self._fail_count = fail_count
        self.call_count = 0

    async def _call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise ConnectionError(f"Flaky failure #{self.call_count}")
        return self._response_text, {"attempt": self.call_count}

    async def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider(name="mock-alpha", response_text="The Northern Lights are caused by solar particles.")


@pytest.fixture
def mock_provider_b() -> MockProvider:
    return MockProvider(name="mock-beta", response_text="Aurora borealis occurs when charged particles from the sun interact with Earth's magnetic field.")


@pytest.fixture
def mock_provider_bad() -> MockProvider:
    """A provider that gives a very different (outlier) response."""
    return MockProvider(name="mock-bad", response_text="Bananas are yellow fruits.")


@pytest.fixture
def mock_unavailable() -> MockProvider:
    return MockProvider(name="mock-offline", available=False)


@pytest.fixture
def sample_responses() -> list[ModelResponse]:
    """A set of realistic model responses for clustering / peer review tests."""
    return [
        ModelResponse(
            provider_name="mock-alpha",
            prompt_variant="original",
            text="The Northern Lights, or aurora borealis, are caused by charged particles from the sun colliding with atmospheric gases.",
            latency=1.0,
        ),
        ModelResponse(
            provider_name="mock-alpha",
            prompt_variant="step_by_step",
            text="Step 1: The sun emits charged particles. Step 2: These particles travel to Earth. Step 3: They collide with nitrogen and oxygen in the atmosphere, producing colorful light.",
            latency=1.2,
        ),
        ModelResponse(
            provider_name="mock-beta",
            prompt_variant="original",
            text="Aurora borealis is a natural light display caused by the interaction of solar wind with the Earth's magnetosphere.",
            latency=0.9,
        ),
        ModelResponse(
            provider_name="mock-beta",
            prompt_variant="step_by_step",
            text="The sun releases solar wind. This wind carries charged particles to Earth. Earth's magnetic field channels them to the poles. The particles excite atmospheric molecules, which emit photons.",
            latency=1.1,
        ),
    ]

