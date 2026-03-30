from __future__ import annotations

import math

import numpy as np

from cce.models.types import Chunk, ScoredChunk
from cce.pipeline.chunker import extract_keywords
from cce.settings import Settings


class ImportanceScorer:
    """Assigns a composite importance score to each chunk.

    Score = w_relevance * cosine_sim(chunk, query)
          + w_recency   * recency_decay(turns_ago)
          + w_keyword   * keyword_jaccard(chunk, query)

    All weights are read from Settings so they can be tuned via .env.
    """

    def __init__(self, settings: Settings):
        self._w_relevance = settings.score_weight_relevance
        self._w_recency = settings.score_weight_recency
        self._w_keyword = settings.score_weight_keyword
        self._decay = settings.recency_decay_rate

    def score(
        self,
        chunks: list[Chunk],
        query_embedding: list[float],
        query_text: str,
        current_turn_index: int,
    ) -> list[ScoredChunk]:
        """Score all chunks against a query. Chunks must have embeddings set."""
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_keywords = extract_keywords(query_text)

        scored: list[ScoredChunk] = []
        for chunk in chunks:
            relevance = self._relevance(chunk, q_vec)
            recency = self._recency(chunk, current_turn_index)
            keyword = self._keyword(chunk, q_keywords)
            composite = (
                self._w_relevance * relevance
                + self._w_recency * recency
                + self._w_keyword * keyword
            )
            scored.append(ScoredChunk(
                chunk=chunk,
                recency_score=recency,
                relevance_score=relevance,
                keyword_score=keyword,
                composite_score=round(composite, 6),
            ))

        return sorted(scored, key=lambda s: s.composite_score, reverse=True)

    # ------------------------------------------------------------------
    # Signal implementations
    # ------------------------------------------------------------------

    def _relevance(self, chunk: Chunk, q_vec: np.ndarray) -> float:
        """Cosine similarity between chunk embedding and query embedding.

        Both vectors are L2-normalised by the embedding provider, so
        cosine similarity = dot product.
        Returns 0.0 if the chunk has no embedding yet.
        """
        if chunk.embedding is None:
            return 0.0
        c_vec = np.array(chunk.embedding, dtype=np.float32)
        score = float(np.dot(c_vec, q_vec))
        # Clamp to [0, 1] — negative similarity treated as no relevance
        return max(0.0, min(1.0, score))

    def _recency(self, chunk: Chunk, current_turn_index: int) -> float:
        """Exponential decay: 1 / (1 + decay_rate * turns_ago).

        A chunk from the current turn scores 1.0; older chunks decay smoothly.
        """
        turns_ago = max(0, current_turn_index - chunk.turn_index)
        return 1.0 / (1.0 + self._decay * turns_ago)

    def _keyword(self, chunk: Chunk, q_keywords: frozenset[str]) -> float:
        """Jaccard similarity between chunk keywords and query keywords.

        Returns 0.0 if either set is empty.
        """
        if not q_keywords:
            return 0.0
        c_keywords = extract_keywords(chunk.content)
        if not c_keywords:
            return 0.0
        intersection = len(c_keywords & q_keywords)
        union = len(c_keywords | q_keywords)
        return intersection / union if union > 0 else 0.0
