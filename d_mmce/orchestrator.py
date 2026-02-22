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
        review_provider_name: str = "openai",
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

        # --- Providers ---
        if providers is not None:
            self._providers = list(providers)
        else:
            self._providers = ProviderFactory.create_all(
                **(provider_configs or {})
            )

        if not self._providers:
            raise RuntimeError(
                "No providers available. Register at least one provider or "
                "pass pre-built instances."
            )

        # --- Review provider ---
        review_provider = next(
            (p for p in self._providers if p.name == review_provider_name),
            self._providers[0],
        )

        # --- Pipeline components ---
        self._perturbator = PromptPerturbator(event_bus=self._bus)
        self._peer_reviewer = PeerReviewer(
            review_provider=review_provider, event_bus=self._bus
        )
        self._clusterer = SemanticClusterer(
            model_name=embedding_model, event_bus=self._bus
        )
        self._meta_judge = MetaJudge(
            providers=self._providers,
            clusterer=self._clusterer,
            stability_threshold=stability_threshold,
            max_reruns=max_stability_reruns,
            event_bus=self._bus,
        )

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
        if not available:
            logger.warning("No providers passed health-check; using all.")
            return list(self._providers)
        return available

    async def _parallel_inference(
        self,
        providers: Sequence[ModelProvider],
        variants: dict[str, str],
    ) -> list[ModelResponse]:
        """Fan out all (provider × variant) calls via ``asyncio.gather()``.

        This is the core async orchestration step.  For *P* providers and
        *V* variants, it launches *P × V* concurrent tasks.
        """
        tasks: list[asyncio.Task[ModelResponse]] = []
        for provider in providers:
            for variant_name, prompt_text in variants.items():
                tasks.append(
                    asyncio.create_task(
                        provider.generate(prompt_text, variant=variant_name)
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: list[ModelResponse] = []
        for r in results:
            if isinstance(r, ModelResponse):
                responses.append(r)
                if self._bus:
                    self._bus.publish(
                        Event(
                            EventType.MODEL_RESPONSE,
                            message=f"Got response from {r.provider_name}::{r.prompt_variant} "
                            f"({len(r.text)} chars, {r.latency:.2f}s)",
                        )
                    )
            else:
                logger.warning("Inference task failed: %s", r)

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

        # 1. Diversity injection
        perturbed = self._perturbator.perturb(query)
        variants = perturbed.variants()
        audit_trail.append(f"Generated {len(variants)} prompt variants.")

        # 2. Filter available providers
        providers = await self._filter_available()
        audit_trail.append(
            f"Available providers: {[p.name for p in providers]}"
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

