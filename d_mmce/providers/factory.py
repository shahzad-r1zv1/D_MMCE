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

        Raises
        ------
        KeyError
            If *name* has not been registered.
        """
        if name not in _REGISTRY:
            raise KeyError(
                f"Unknown provider '{name}'. "
                f"Available: {ProviderFactory.available_names()}"
            )
        return _REGISTRY[name](**kwargs)

    @staticmethod
    def create_all(**provider_configs: dict[str, Any]) -> list[ModelProvider]:
        """Create all registered providers, passing per-provider kwargs.

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
            cfg = provider_configs.get(name, {})
            try:
                providers.append(cls(**cfg))
            except Exception:
                logger.warning("Skipping provider '%s' (instantiation failed)", name, exc_info=True)
        return providers

