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
import random
import time
import uuid
from typing import Any, Sequence

from d_mmce.meta_judge import MetaJudge
from d_mmce.observer import Event, EventBus, EventType, LoggingObserver
from d_mmce.peer_reviewer import PeerReviewer
from d_mmce.prompt_perturbator import PromptPerturbator
from d_mmce.providers.base import ModelProvider
from d_mmce.providers.factory import ProviderFactory
from d_mmce.schemas import FailureCategory, FinalVerdict, ModelResponse
from d_mmce.semantic_clusterer import SemanticClusterer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Failure classification                                             #
# ------------------------------------------------------------------ #

def classify_failure(exc: Exception) -> FailureCategory:
    """Categorise an exception into a retry-relevant failure bucket."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    # Timeout
    if "timeout" in name or "timeout" in msg:
        return FailureCategory.TIMEOUT
    if isinstance(exc, asyncio.TimeoutError):
        return FailureCategory.TIMEOUT

    # Auth
    if "auth" in name or "auth" in msg or "401" in msg or "403" in msg:
        return FailureCategory.AUTH
    if "permission" in msg or "api key" in msg or "invalid key" in msg:
        return FailureCategory.AUTH

    # Quota / rate-limit
    if "429" in msg or "rate" in msg or "quota" in msg or "limit" in msg:
        return FailureCategory.QUOTA

    # Non-retryable
    if "400" in msg or "404" in msg or "invalid" in name:
        return FailureCategory.PERMANENT

    return FailureCategory.TRANSIENT


def _is_retryable(category: FailureCategory) -> bool:
    """Return ``True`` if the failure category warrants a retry."""
    return category in (
        FailureCategory.TIMEOUT,
        FailureCategory.QUOTA,
        FailureCategory.TRANSIENT,
    )


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
        # --- Concurrency & Retry (Feature 4) ---
        max_concurrent_tasks: int = 4,
        max_retries: int = 2,
        retry_base_delay: float = 2.0,
        request_timeout: float = 120.0,
        # --- Streaming (Feature 3) ---
        enable_streaming: bool = True,
        # --- History (Feature 5) ---
        history_db: Any | None = None,
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

        # --- Concurrency & Retry ---
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._request_timeout = request_timeout

        # --- Streaming ---
        self._enable_streaming = enable_streaming

        # --- History ---
        self._history_db = history_db

        # --- Run tracking ---
        self._current_run_id = ""

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
            review_provider=review_provider,
            event_bus=self._bus,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
            request_timeout=self._request_timeout,
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

    async def _call_with_retry(
        self,
        provider: ModelProvider,
        prompt: str,
        variant: str,
    ) -> ModelResponse:
        """Call a provider with semaphore gating, streaming, timeout, and jittered exponential backoff retry."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                async with self._semaphore:
                    if self._enable_streaming and provider.supports_streaming:
                        def _on_token(token: str) -> None:
                            if self._bus:
                                self._bus.publish(
                                    Event(
                                        EventType.TOKEN_CHUNK,
                                        message="",
                                        payload={
                                            "provider": provider.name,
                                            "variant": variant,
                                            "token": token,
                                        },
                                        run_id=self._current_run_id,
                                    )
                                )
                        coro = provider.generate_stream(
                            prompt, variant=variant, on_token=_on_token
                        )
                    else:
                        coro = provider.generate(prompt, variant=variant)

                    return await asyncio.wait_for(coro, timeout=self._request_timeout)
            except Exception as exc:
                last_exc = exc
                category = classify_failure(exc)

                if attempt < self._max_retries and _is_retryable(category):
                    # Jittered exponential backoff
                    jitter = random.uniform(0, self._retry_base_delay)
                    delay = self._retry_base_delay * (2 ** attempt) + jitter
                    logger.warning(
                        "Retry %d/%d for %s::%s — %s [%s] (delay %.1fs)",
                        attempt + 1, self._max_retries, provider.name,
                        variant, type(exc).__name__, category.name, delay,
                    )
                    await asyncio.sleep(delay)
                elif not _is_retryable(category):
                    logger.warning(
                        "Non-retryable failure for %s::%s — %s [%s]",
                        provider.name, variant,
                        type(exc).__name__, category.name,
                    )
                    break

        raise last_exc  # type: ignore[misc]

    async def _parallel_inference(
        self,
        providers: Sequence[ModelProvider],
        variants: dict[str, str],
    ) -> list[ModelResponse]:
        """Fan out all (provider × variant) calls with concurrency control.

        Uses a semaphore to limit parallelism, exponential-backoff retry
        for transient failures, and per-token streaming for providers
        that support it.  Events are published as each task completes.
        """
        tasks: list[asyncio.Task[ModelResponse]] = []
        for provider in providers:
            for variant_name, prompt_text in variants.items():
                tasks.append(
                    asyncio.create_task(
                        self._call_with_retry(provider, prompt_text, variant_name)
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
                            run_id=self._current_run_id,
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
            The globally optimal answer, stability score, confidence score,
            stage timings, and audit trail.

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
        stage_timings: dict[str, float] = {}

        # Generate a unique run ID for this pipeline execution
        self._current_run_id = uuid.uuid4().hex[:12]
        logger.info("Pipeline run_id=%s started for query=%r",
                     self._current_run_id, query[:80])

        # 0. Ensure providers are resolved (auto-discovery if needed)
        await self._ensure_providers()

        # 1. Diversity injection
        t0 = time.perf_counter()
        perturbed = self._perturbator.perturb(query)
        variants = perturbed.variants()
        stage_timings["diversify"] = round(time.perf_counter() - t0, 4)
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
                run_id=self._current_run_id,
                stage_timings=stage_timings,
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
            review_provider=review_provider,
            event_bus=self._bus,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
            request_timeout=self._request_timeout,
        )
        self._meta_judge = MetaJudge(
            providers=providers,
            clusterer=self._clusterer,
            stability_threshold=self._stability_threshold,
            max_reruns=self._max_stability_reruns,
            event_bus=self._bus,
        )

        # 3. Parallel inference via asyncio.gather()
        t0 = time.perf_counter()
        responses = await self._parallel_inference(providers, variants)
        stage_timings["infer"] = round(time.perf_counter() - t0, 4)
        audit_trail.append(
            f"Collected {len(responses)} responses from "
            f"{len(providers)} providers × {len(variants)} variants."
        )

        if not responses:
            return FinalVerdict(
                answer="ERROR: No responses received from any provider.",
                stability_score=0.0,
                run_id=self._current_run_id,
                stage_timings=stage_timings,
                audit_trail=audit_trail,
            )

        # 4. Peer review (cross-examination)
        t0 = time.perf_counter()
        matrix = await self._peer_reviewer.review(responses, providers)
        stage_timings["review"] = round(time.perf_counter() - t0, 4)
        audit_trail.append(
            f"Peer review complete: {len(matrix.critiques)} critiques generated."
        )

        # 5. Semantic clustering
        t0 = time.perf_counter()
        cluster = self._clusterer.cluster(responses)
        stage_timings["cluster"] = round(time.perf_counter() - t0, 4)
        audit_trail.append(
            f"Consensus cluster: {len(cluster.member_responses)} members, "
            f"{len(cluster.outliers)} outliers discarded."
        )

        # Handle insufficient consensus
        if cluster.insufficient_consensus:
            audit_trail.append(
                "WARNING: Insufficient consensus — cluster too small or ratio "
                f"too low ({cluster.consensus_ratio:.2f})."
            )
            return FinalVerdict(
                answer=(
                    "INSUFFICIENT CONSENSUS: The models could not reach "
                    "agreement. The responses were too diverse to form a "
                    "reliable consensus cluster."
                ),
                stability_score=0.0,
                confidence_score=0.0,
                run_id=self._current_run_id,
                stage_timings=stage_timings,
                audit_trail=audit_trail,
            )

        # 6. Meta-Judge synthesis + Stability Loop
        t0 = time.perf_counter()
        verdict = await self._meta_judge.synthesize(cluster, matrix)
        stage_timings["synthesize"] = round(time.perf_counter() - t0, 4)
        verdict.audit_trail = audit_trail + verdict.audit_trail
        verdict.run_id = self._current_run_id
        verdict.stage_timings = stage_timings

        # 7. Compute composite confidence score
        total_responses = len(cluster.member_responses) + len(cluster.outliers)
        consensus_strength = (
            len(cluster.member_responses) / total_responses
            if total_responses > 0 else 0.0
        )
        avg_penalty = (
            sum(matrix.scores.values()) / len(matrix.scores)
            if matrix.scores else 0.0
        )
        contradiction_factor = 1.0 - avg_penalty
        stability = verdict.stability_score

        verdict.confidence_score = round(
            0.4 * consensus_strength
            + 0.3 * contradiction_factor
            + 0.3 * stability,
            4,
        )

        if self._bus:
            self._bus.publish(
                Event(
                    EventType.FINAL_VERDICT,
                    message=(
                        f"Final verdict: stability={verdict.stability_score:.4f}, "
                        f"confidence={verdict.confidence_score:.4f}, "
                        f"reruns={verdict.num_reruns}"
                    ),
                    payload={
                        "answer": verdict.answer,
                        "stability_score": verdict.stability_score,
                        "confidence_score": verdict.confidence_score,
                        "num_reruns": verdict.num_reruns,
                        "audit_trail": verdict.audit_trail,
                        "stage_timings": verdict.stage_timings,
                    },
                    run_id=self._current_run_id,
                )
            )

        # 8. Persist to history database (Feature 5)
        if self._history_db:
            try:
                await self._history_db.save_run(
                    query=query,
                    verdict=verdict,
                    responses=responses,
                    critiques=matrix.critiques,
                )
            except Exception:
                logger.warning("Failed to save run to history DB", exc_info=True)

        return verdict

