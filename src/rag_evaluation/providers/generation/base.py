from abc import ABC, abstractmethod

from pydantic import BaseModel

from rag_evaluation.models import LLMProviderResult


class LLMProvider(ABC):
    @abstractmethod
    def call(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int = -1,
    ) -> LLMProviderResult: ...
