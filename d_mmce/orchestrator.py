"""
D-MMCE Orchestrator – The Async Pipeline
==========================================
The :class:`D_MMCE` class wires every stage together and exposes a single
``await engine.run(query)`` entry point.

Pipeline
--------
1. **Diversity Injection** — :class:`PromptPerturbator` generates 4 prompt
   variants (original + 3 perturbations).
2. **Parallel Inference** — ``asyncio.gather()`` fans out *providers × variants*
   calls concurrently, producing *N × 4* responses.
3. **Cross-Examination** — :class:`PeerReviewer` generates a
   :class:`ContradictionMatrix` via pairwise critique.
4. **Semantic Clustering** — :class:`SemanticClusterer` embeds all responses,
   finds the densest cluster, and discards outliers.
5. **Meta-Judge Synthesis** — :class:`MetaJudge` synthesises the final answer
   from the consensus cluster and runs the **Stability Loop**.

Global vs. Local Optimality
----------------------------
* A **Local Optimum** is an individual model response that may be plausible
  in isolation but diverges from the ensemble consensus (it lands outside
  the main semantic cluster) or is contradicted by peer reviewers.
* The **Global Optimum** is the synthesised answer that:
  - belongs to the densest semantic cluster (maximum agreement),
  - survives peer review with low contradiction penalties,
  - remains **stable** across successive synthesis rounds (Stability Loop).

The Stability Loop quantifies convergence: if re-generating the synthesis
yields a semantically similar output (cosine sim ≥ threshold), the system
has converged and the answer is declared globally optimal.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from d_mmce.meta_judge import MetaJudge
from d_mmce.observer import Event, EventBus, EventType, LoggingObserver
from d_mmce.peer_reviewer import PeerReviewer
from d_mmce.prompt_perturbator import PromptPerturbator
from d_mmce.providers.base import ModelProvider
from d_mmce.providers.factory import ProviderFactory
from d_mmce.schemas import FinalVerdict, ModelResponse
from d_mmce.semantic_clusterer import SemanticClusterer

logger = logging.getLogger(__name__)


class D_MMCE:
    """Dynamic Multi-Model Consensus Engine.

    Aggregates "weak" LLM learners to find a **Globally Optimal** output
    by combining diversity injection, parallel inference, peer review,
    semantic clustering, and a stability-checked meta-synthesis.

    Parameters
    ----------
    providers : Sequence[ModelProvider], optional
        Pre-built provider instances.  If ``None``, the factory creates
        all registered providers using *provider_configs*.
    provider_configs : dict[str, dict], optional
        Per-provider kwargs passed to :meth:`ProviderFactory.create_all`.
    review_provider_name : str
        Name of the provider used for peer reviews (default ``"openai"``).
    embedding_model : str
        ``sentence-transformers`` model for clustering (default
        ``"all-MiniLM-L6-v2"``).
    stability_threshold : float
        Cosine-similarity threshold for the Stability Loop (default ``0.85``).
    max_stability_reruns : int
        Maximum synthesis re-runs (default ``3``).
    enable_logging_observer : bool
        Attach a :class:`LoggingObserver` to the event bus (default ``True``).
    """

    def __init__(
        self,
        providers: Sequence[ModelProvider] | None = None,
        provider_configs: dict[str, dict[str, Any]] | None = None,
        review_provider_name: str = "auto",
        embedding_model: str = "all-MiniLM-L6-v2",
        stability_threshold: float = 0.85,
        max_stability_reruns: int = 3,
        enable_logging_observer: bool = True,
    ) -> None:
        # --- Event bus ---
        self._bus = EventBus()
        if enable_logging_observer:
            observer = LoggingObserver()
            self._bus.subscribe_all(observer)

        # --- Providers (may be lazily populated in run()) ---
        self._providers: list[ModelProvider] = list(providers) if providers else []
        self._provider_configs = provider_configs or {}
        self._providers_resolved = providers is not None  # skip auto-discovery if explicit

        # --- Deferred config ---
        self._review_provider_name = review_provider_name
        self._embedding_model = embedding_model
        self._stability_threshold = stability_threshold
        self._max_stability_reruns = max_stability_reruns

        # --- Eagerly build components if providers were given ---
        if self._providers:
            self._build_components()

    def _build_components(self) -> None:
        """Wire up the pipeline components once providers are resolved."""
        if self._review_provider_name == "auto":
            review_provider = self._providers[0]
        else:
            review_provider = next(
                (p for p in self._providers if p.name == self._review_provider_name),
                self._providers[0],
            )

        self._perturbator = PromptPerturbator(event_bus=self._bus)
        self._peer_reviewer = PeerReviewer(
            review_provider=review_provider, event_bus=self._bus
        )
        self._clusterer = SemanticClusterer(
            model_name=self._embedding_model, event_bus=self._bus
        )
        self._meta_judge = MetaJudge(
            providers=self._providers,
            clusterer=self._clusterer,
            stability_threshold=self._stability_threshold,
            max_reruns=self._max_stability_reruns,
            event_bus=self._bus,
        )

    async def _ensure_providers(self) -> None:
        """Auto-discover providers if none were supplied."""
        if self._providers_resolved:
            return

        self._providers_resolved = True
        self._providers = await ProviderFactory.create_all_async(
            **self._provider_configs
        )

        if not self._providers:
            raise RuntimeError(
                "No providers available.  Either:\n"
                "  • Set API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY) in .env\n"
                "  • Start Ollama and pull a model: ollama pull llama3.2\n"
                "  • Pass providers explicitly to D_MMCE(providers=[...])"
            )

        logger.info("Auto-discovered providers: %s", [p.name for p in self._providers])
        self._build_components()

    @property
    def event_bus(self) -> EventBus:
        """Expose the bus so callers can subscribe to pipeline events."""
        return self._bus

    # ------------------------------------------------------------------ #
    #  Stage helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _filter_available(self) -> list[ModelProvider]:
        """Return only providers that pass their health-check."""
        checks = await asyncio.gather(
            *(p.is_available() for p in self._providers),
            return_exceptions=True,
        )
        available = [
            p
            for p, ok in zip(self._providers, checks)
            if ok is True
        ]
        unavailable = [
            p.name
            for p, ok in zip(self._providers, checks)
            if ok is not True
        ]
        if unavailable:
            logger.info("Unavailable providers (skipped): %s", unavailable)
        if not available:
            logger.warning(
                "No providers passed health-check. "
                "Set API keys or start Ollama with a pulled model."
            )
        return available

    async def _parallel_inference(
        self,
        providers: Sequence[ModelProvider],
        variants: dict[str, str],
    ) -> list[ModelResponse]:
        """Fan out all (provider × variant) calls via ``asyncio.gather()``.

        This is the core async orchestration step.  For *P* providers and
        *V* variants, it launches *P × V* concurrent tasks.  Events are
        published **as each task completes** so the UI can stream responses
        live rather than waiting for the slowest model.
        """
        tasks: list[asyncio.Task[ModelResponse]] = []
        for provider in providers:
            for variant_name, prompt_text in variants.items():
                tasks.append(
                    asyncio.create_task(
                        provider.generate(prompt_text, variant=variant_name)
                    )
                )

        responses: list[ModelResponse] = []
        failures: list[str] = []

        for coro in asyncio.as_completed(tasks):
            try:
                r = await coro
                responses.append(r)
                if self._bus:
                    self._bus.publish(
                        Event(
                            EventType.MODEL_RESPONSE,
                            message=f"Got response from {r.provider_name}::{r.prompt_variant} "
                            f"({len(r.text)} chars, {r.latency:.2f}s)",
                            payload={
                                "provider": r.provider_name,
                                "variant": r.prompt_variant,
                                "text": r.text,
                                "latency": round(r.latency, 2),
                                "chars": len(r.text),
                            },
                        )
                    )
            except Exception as exc:
                err_name = type(exc).__name__
                err_msg = str(exc)[:120]
                failures.append(f"{err_name}: {err_msg}")
                logger.warning("Inference task failed: %s: %s", err_name, err_msg)

        if failures and not responses:
            logger.error(
                "All %d inference tasks failed. Sample errors:\n  %s",
                len(failures),
                "\n  ".join(failures[:5]),
            )

        return responses

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    async def run(self, query: str) -> FinalVerdict:
        """Execute the full D-MMCE pipeline.

        Parameters
        ----------
        query : str
            The user's question or task.

        Returns
        -------
        FinalVerdict
            The globally optimal answer, stability score, and audit trail.

        How the pipeline distinguishes Local from Global Optimality
        -----------------------------------------------------------
        1. **Diversity Injection** forces each model to reason from multiple
           angles, reducing the chance that a single framing biases the
           ensemble toward a local optimum.
        2. **Parallel Inference** collects a broad sample of *P × V*
           responses — the "population" in optimisation terms.
        3. **Peer Review** penalises responses with verified hallucinations,
           acting as a fitness function that down-weights local optima.
        4. **Semantic Clustering** mathematically identifies the densest
           region in embedding space.  Outliers (local optima) are discarded.
        5. **Meta-Judge Synthesis** produces a single answer from the
           consensus cluster.
        6. **Stability Loop** re-generates the synthesis and checks cosine
           similarity.  Convergence (similarity ≥ threshold) signals that
           the answer is a **Global Optimum** — it is robust to re-sampling.
           Divergence flags the answer as a potential **Local Optimum**.
        """
        audit_trail: list[str] = []

        # 0. Ensure providers are resolved (auto-discovery if needed)
        await self._ensure_providers()

        # 1. Diversity injection
        perturbed = self._perturbator.perturb(query)
        variants = perturbed.variants()
        audit_trail.append(f"Generated {len(variants)} prompt variants.")

        # 2. Filter available providers
        providers = await self._filter_available()
        audit_trail.append(
            f"Available providers: {[p.name for p in providers]}"
        )

        if not providers:
            return FinalVerdict(
                answer="ERROR: No providers available. Configure API keys or start Ollama.",
                stability_score=0.0,
                audit_trail=audit_trail,
            )

        # Rebuild peer-reviewer and meta-judge with only available providers
        # so the Meta-Judge never picks an unreachable model.
        if self._review_provider_name == "auto":
            review_provider = providers[0]
        else:
            review_provider = next(
                (p for p in providers if p.name == self._review_provider_name),
                providers[0],
            )
        logger.info("Review provider: %s | Meta-Judge pool: %s",
                     review_provider.name, [p.name for p in providers])
        self._peer_reviewer = PeerReviewer(
            review_provider=review_provider, event_bus=self._bus
        )
        self._meta_judge = MetaJudge(
            providers=providers,
            clusterer=self._clusterer,
            stability_threshold=self._stability_threshold,
            max_reruns=self._max_stability_reruns,
            event_bus=self._bus,
        )

        # 3. Parallel inference via asyncio.gather()
        responses = await self._parallel_inference(providers, variants)
        audit_trail.append(
            f"Collected {len(responses)} responses from "
            f"{len(providers)} providers × {len(variants)} variants."
        )

        if not responses:
            return FinalVerdict(
                answer="ERROR: No responses received from any provider.",
                stability_score=0.0,
                audit_trail=audit_trail,
            )

        # 4. Peer review (cross-examination)
        matrix = await self._peer_reviewer.review(responses, providers)
        audit_trail.append(
            f"Peer review complete: {len(matrix.critiques)} critiques generated."
        )

        # 5. Semantic clustering
        cluster = self._clusterer.cluster(responses)
        audit_trail.append(
            f"Consensus cluster: {len(cluster.member_responses)} members, "
            f"{len(cluster.outliers)} outliers discarded."
        )

        # 6. Meta-Judge synthesis + Stability Loop
        verdict = await self._meta_judge.synthesize(cluster, matrix)
        verdict.audit_trail = audit_trail + verdict.audit_trail

        if self._bus:
            self._bus.publish(
                Event(
                    EventType.FINAL_VERDICT,
                    message=(
                        f"Final verdict: stability={verdict.stability_score:.4f}, "
                        f"reruns={verdict.num_reruns}"
                    ),
                )
            )

        return verdict

