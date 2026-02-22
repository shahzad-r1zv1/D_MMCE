"""
Provider Factory – Registration & Creation
===========================================
Uses a class-level registry so each concrete provider can self-register
with a ``@register("name")`` decorator.  The orchestrator calls
``ProviderFactory.create("openai", api_key="…")`` without importing
concrete classes.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from d_mmce.providers.base import ModelProvider

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[ModelProvider]] = {}


def register(name: str):
    """Class decorator that registers a :class:`ModelProvider` subclass."""

    def decorator(cls: Type[ModelProvider]):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


class ProviderFactory:
    """Factory for constructing :class:`ModelProvider` instances by name."""

    @staticmethod
    def available_names() -> list[str]:
        """Return the names of all registered providers."""
        return list(_REGISTRY.keys())

    @staticmethod
    def create(name: str, **kwargs: Any) -> ModelProvider:
        """Instantiate a registered provider.

        Supports the ``ollama:<model_tag>`` shorthand — e.g.
        ``ProviderFactory.create("ollama:mistral")`` creates an
        :class:`OllamaProvider` targeting the ``mistral`` model.

        Raises
        ------
        KeyError
            If *name* has not been registered (and is not an ``ollama:``
            prefixed shorthand).
        """
        # Handle "ollama:<model>" shorthand
        if name.startswith("ollama:") and "ollama" in _REGISTRY:
            model_tag = name[len("ollama:"):]
            if model_tag:
                kwargs.setdefault("model", model_tag)
                return _REGISTRY["ollama"](**kwargs)

        if name not in _REGISTRY:
            raise KeyError(
                f"Unknown provider '{name}'. "
                f"Available: {ProviderFactory.available_names()}"
            )
        return _REGISTRY[name](**kwargs)

    @staticmethod
    def create_all(**provider_configs: dict[str, Any]) -> list[ModelProvider]:
        """Create all registered providers, passing per-provider kwargs.

        .. note::
            The ``ollama`` provider is **skipped** by this synchronous method
            because discovering locally-available models requires async I/O.
            Use :meth:`create_all_async` instead to include local models.

        Parameters
        ----------
        **provider_configs
            Mapping of ``provider_name → {kwarg: value, …}``.  Providers
            whose name is absent from *provider_configs* are created with
            no arguments.

        Returns
        -------
        list[ModelProvider]
        """
        providers: list[ModelProvider] = []
        for name, cls in _REGISTRY.items():
            if name == "ollama":
                # Ollama needs async discovery — handled by create_all_async
                continue
            cfg = provider_configs.get(name, {})
            try:
                providers.append(cls(**cfg))
            except Exception:
                logger.warning("Skipping provider '%s' (instantiation failed)", name, exc_info=True)
        return providers

    @staticmethod
    async def create_all_async(**provider_configs: dict[str, Any]) -> list[ModelProvider]:
        """Create all providers including auto-discovered Ollama local models.

        This async variant queries the Ollama server for available models
        and creates one :class:`OllamaProvider` per installed model.

        Returns
        -------
        list[ModelProvider]
        """
        providers = ProviderFactory.create_all(**provider_configs)

        # Auto-discover Ollama models
        if "ollama" in _REGISTRY:
            try:
                from d_mmce.providers.ollama_provider import OllamaProvider
                models = await OllamaProvider.list_local_models()
                for m in models:
                    full_name = m.get("name", "")
                    if full_name:
                        providers.append(OllamaProvider(model=full_name))
                        logger.info("Auto-discovered local model: ollama:%s", full_name)
                if not models:
                    logger.info("Ollama is running but no models installed.")
            except Exception:
                logger.debug("Ollama auto-discovery failed", exc_info=True)

        return providers

