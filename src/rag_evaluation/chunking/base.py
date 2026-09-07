from abc import ABC, abstractmethod

from rag_evaluation.models import Chunk, Document


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]: ...
