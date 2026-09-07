from abc import ABC, abstractmethod

from rag_evaluation.models import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self) -> list[Document]: ...
