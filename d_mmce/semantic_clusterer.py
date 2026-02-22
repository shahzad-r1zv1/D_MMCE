"""
Semantic Clusterer – Consensus Aggregator
==========================================
Encodes all model responses into a shared embedding space using
``sentence-transformers`` and then finds the **Consensus Cluster** –
the densest group of semantically similar answers.

Algorithm
---------
1. Embed all response texts with ``all-MiniLM-L6-v2``.
2. Try **HDBSCAN** to find natural density-based clusters.
3. If HDBSCAN produces only noise (no cluster), fall back to a simpler
   **cosine-similarity threshold** approach.
4. Select the largest cluster as the *Consensus Cluster*.
5. Compute the centroid of that cluster and pick the response whose
   embedding is closest to the centroid — this is the **Candidate Winner**.
6. Remaining responses are labelled *outliers* (Local Optima).

Mathematical Justification
--------------------------
The *Global Optimum* in semantic space is defined as the point that
minimises the sum of squared distances to the cluster members — i.e.,
the **centroid**.  By selecting the real response closest to this centroid,
we avoid fabricating new text while still anchoring the answer to the
point of maximum ensemble agreement.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from d_mmce.observer import Event, EventBus, EventType
from d_mmce.schemas import ConsensusCluster, ModelResponse

logger = logging.getLogger(__name__)


def _cosine_similarity(a: NDArray, b: NDArray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm > 0 else 0.0


class SemanticClusterer:
    """Find the consensus cluster among model responses.

    Parameters
    ----------
    model_name : str
        ``sentence-transformers`` model to use for encoding.
    similarity_threshold : float
        Cosine-similarity threshold for the fallback clustering method.
    event_bus : EventBus, optional
        Publish clustering events.
    """

    _encoder = None  # lazy-loaded class-level cache

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.65,
        event_bus: EventBus | None = None,
    ) -> None:
        self._model_name = model_name
        self._sim_threshold = similarity_threshold
        self._bus = event_bus

    def _get_encoder(self):
        """Lazy-load the SentenceTransformer model (expensive, ~2 s)."""
        if SemanticClusterer._encoder is None:
            from sentence_transformers import SentenceTransformer

            SemanticClusterer._encoder = SentenceTransformer(self._model_name)
        return SemanticClusterer._encoder

    def _embed(self, texts: list[str]) -> NDArray:
        """Return an (N, D) matrix of embeddings."""
        encoder = self._get_encoder()
        return encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # ------------------------------------------------------------------ #
    #  Clustering strategies                                              #
    # ------------------------------------------------------------------ #

    def _cluster_hdbscan(self, embeddings: NDArray) -> NDArray:
        """Try HDBSCAN clustering.  Returns label array (-1 = noise)."""
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(embeddings)
            return np.asarray(labels)
        except Exception:
            logger.debug("HDBSCAN failed; falling back to cosine threshold", exc_info=True)
            return np.full(len(embeddings), -1)

    def _cluster_cosine_fallback(self, embeddings: NDArray) -> NDArray:
        """Fallback: group responses by pairwise cosine similarity."""
        n = len(embeddings)
        labels = np.full(n, -1)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue
            labels[i] = cluster_id
            for j in range(i + 1, n):
                if labels[j] != -1:
                    continue
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self._sim_threshold:
                    labels[j] = cluster_id
            cluster_id += 1

        return labels

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def cluster(self, responses: Sequence[ModelResponse]) -> ConsensusCluster:
        """Cluster *responses* and return the consensus cluster.

        The response closest to the centroid of the largest cluster is
        chosen as the *Candidate Winner*.  Responses outside this cluster
        are labelled as outliers (Local Optima).

        Parameters
        ----------
        responses : Sequence[ModelResponse]

        Returns
        -------
        ConsensusCluster
        """
        if not responses:
            return ConsensusCluster(centroid_text="")

        texts = [r.text for r in responses]
        embeddings = self._embed(texts)

        # --- Try HDBSCAN first, fall back to cosine threshold ---
        labels = self._cluster_hdbscan(embeddings)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        if not unique_labels:
            # HDBSCAN found no clusters → fallback
            labels = self._cluster_cosine_fallback(embeddings)
            unique_labels = set(labels)
            unique_labels.discard(-1)

        if not unique_labels:
            # Even fallback couldn't cluster → treat all as one cluster
            labels = np.zeros(len(responses), dtype=int)
            unique_labels = {0}

        # --- Pick the largest cluster ---
        cluster_sizes = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
        consensus_label = max(cluster_sizes, key=cluster_sizes.get)  # type: ignore[arg-type]

        members: list[ModelResponse] = []
        member_embeds: list[NDArray] = []
        outliers: list[ModelResponse] = []

        for idx, resp in enumerate(responses):
            if labels[idx] == consensus_label:
                members.append(resp)
                member_embeds.append(embeddings[idx])
            else:
                outliers.append(resp)

        # --- Compute centroid and pick nearest response ---
        centroid = np.mean(member_embeds, axis=0)
        distances = [
            np.linalg.norm(e - centroid) for e in member_embeds
        ]
        winner_idx = int(np.argmin(distances))
        centroid_text = members[winner_idx].text

        # --- Publish events ---
        if self._bus:
            self._bus.publish(
                Event(
                    EventType.CLUSTER_FORMED,
                    message=(
                        f"Consensus cluster has {len(members)} members, "
                        f"{len(outliers)} outliers discarded."
                    ),
                    payload={
                        "cluster_size": len(members),
                        "outlier_count": len(outliers),
                        "winner_provider": members[winner_idx].provider_name,
                    },
                )
            )
            for out in outliers:
                self._bus.publish(
                    Event(
                        EventType.OUTLIER_DISCARDED,
                        message=f"Outlier: {out.provider_name}::{out.prompt_variant}",
                    )
                )

        return ConsensusCluster(
            centroid_text=centroid_text,
            member_responses=members,
            outliers=outliers,
            centroid_embedding=centroid.tolist(),
        )

    def cosine_similarity_texts(self, text_a: str, text_b: str) -> float:
        """Convenience: cosine similarity between two texts."""
        embeds = self._embed([text_a, text_b])
        return _cosine_similarity(embeds[0], embeds[1])

