from abc import ABC, abstractmethod

from rag.models.documents import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self) -> list[Document]: ...
