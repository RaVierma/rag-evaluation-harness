from abc import ABC, abstractmethod

from rag_evaluation.models import LLMProviderResult


class LLMProvider(ABC):
    @abstractmethod
    def call(self, prompt: str) -> LLMProviderResult: ...
