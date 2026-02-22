"""
Providers package – auto-imports all concrete providers to trigger
``@register(...)`` decorators.
"""

from d_mmce.providers.base import ModelProvider
from d_mmce.providers.factory import ProviderFactory, register

# Import concrete providers so they self-register via @register(...)
from d_mmce.providers import (  # noqa: F401
    openai_provider,
    anthropic_provider,
    gemini_provider,
    ollama_provider,
)

__all__ = ["ModelProvider", "ProviderFactory", "register"]

