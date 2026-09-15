from abc import ABC, abstractmethod

from rag.models.chunks import Chunk
from rag.models.documents import Document


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]: ...
