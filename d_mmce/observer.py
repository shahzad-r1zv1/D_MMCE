"""
Observer Pattern – EventBus
===========================
A lightweight publish-subscribe bus used by the **PeerReviewer** and other
pipeline stages to emit structured events (critiques, clustering decisions,
stability loop iterations) without coupling to concrete loggers or UIs.

Subscribers implement the :class:`EventObserver` protocol and register
themselves with the singleton :data:`event_bus`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Categories of pipeline events."""

    PROMPT_PERTURBED = auto()
    MODEL_RESPONSE = auto()
    TOKEN_CHUNK = auto()
    PEER_CRITIQUE = auto()
    CLUSTER_FORMED = auto()
    OUTLIER_DISCARDED = auto()
    SYNTHESIS_STARTED = auto()
    STABILITY_CHECK = auto()
    STABILITY_RERUN = auto()
    FINAL_VERDICT = auto()


@dataclass
class Event:
    """A single pipeline event."""

    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    run_id: str = ""


class EventObserver(Protocol):
    """Protocol that any subscriber must satisfy."""

    def on_event(self, event: Event) -> None: ...


class EventBus:
    """Simple synchronous pub-sub bus.

    Usage::

        bus = EventBus()
        bus.subscribe(EventType.PEER_CRITIQUE, my_logger)
        bus.publish(Event(EventType.PEER_CRITIQUE, message="…"))
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {}

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None] | EventObserver,
    ) -> None:
        """Register *callback* (or an :class:`EventObserver`) for *event_type*."""
        fn = callback if callable(callback) and not hasattr(callback, "on_event") else getattr(callback, "on_event")
        self._subscribers.setdefault(event_type, []).append(fn)

    def subscribe_all(self, callback: Callable[[Event], None] | EventObserver) -> None:
        """Register *callback* for **every** event type."""
        for et in EventType:
            self.subscribe(et, callback)

    def publish(self, event: Event) -> None:
        """Dispatch *event* to all registered subscribers."""
        for fn in self._subscribers.get(event.event_type, []):
            try:
                fn(event)
            except Exception:
                logger.exception("Subscriber raised for %s", event.event_type)


class LoggingObserver:
    """Default observer that writes every event to Python's logging module."""

    def on_event(self, event: Event) -> None:
        logger.info("[%s] %s", event.event_type.name, event.message or event.payload)


# Module-level convenience instance
event_bus = EventBus()

