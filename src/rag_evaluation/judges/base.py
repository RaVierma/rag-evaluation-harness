from abc import ABC, abstractmethod

from rag_evaluation.models import GenerationEvaluation


class Judge(ABC):
    @abstractmethod
    def evaluate(
        self,
        question: str,
        context: str,
        expected_answer: str,
        generated_answer: str,
    ) -> GenerationEvaluation: ...
