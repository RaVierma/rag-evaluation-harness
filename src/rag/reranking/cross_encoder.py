from sentence_transformers import CrossEncoder

from rag.models.chunks import HybridRetrievedChunk, RerankedChunk
from rag.reranking.base import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[HybridRetrievedChunk],
        top_k: int,
    ) -> list[RerankedChunk]:

        if not query.strip():
            raise ValueError("query must not be empty")

        if not candidates:
            return []

        if top_k <= 0:
            raise ValueError("top k must be greater than zero")

        pairs = [(query, candidate.chunk.content) for candidate in candidates]

        scores = self._model.predict(pairs)

        reranked_chunks = [
            RerankedChunk(
                chunk=candidate,
                rerank_score=float(score),
            )
            for candidate, score in zip(candidates, scores)
        ]

        reranked_chunks = sorted(
            reranked_chunks,
            key=lambda item: item.rerank_score,
            reverse=True,
        )

        return reranked_chunks[:top_k]
