from abc import ABC, abstractmethod

from rag_evaluation.models import HybridRetrievedChunk, RerankedChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[HybridRetrievedChunk],
        top_k: int,
    ) -> list[RerankedChunk]: ...
