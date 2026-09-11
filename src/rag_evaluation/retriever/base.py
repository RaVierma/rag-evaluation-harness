from abc import ABC, abstractmethod

from rag_evaluation.models.chunks import EmbeddedChunk, RetrievedChunk


class Retriever(ABC):
    def __init__(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ):
        self._embedded_chunks = embedded_chunks

    @abstractmethod
    def retrieve(
        self,
        query_embedding: tuple[float, ...],
        candidate_k: int,
    ) -> list[RetrievedChunk]: ...
