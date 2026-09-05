import json

from pydantic_core import ValidationError

from rag_evaluation.judges.base import Judge
from rag_evaluation.models import GenerationEvaluation
from rag_evaluation.providers.base import LLMProvider
from rag_evaluation.utils.helper import load_builder


class DummyJudge(Judge):
    def __init__(self, provider: LLMProvider):
        super().__init__()
        self.provider = provider

    def evaluate(
        self,
        question: str,
        context: str,
        expected_answer: str,
        generated_answer: str,
    ) -> GenerationEvaluation:

        if not all(
            [
                question,
                expected_answer,
                generated_answer,
            ]
        ):
            raise ValueError(
                "question, expected_answer, and generated_answer are required"
            )

        if not context:
            context = ""

        prompt = load_builder(
            "v1", question, context, expected_answer, generated_answer
        )

        provider_result = self.provider.call(prompt)

        if provider_result.success:
            try:
                evaluation_data = json.loads(provider_result.content)
                return GenerationEvaluation(**evaluation_data)
            except json.JSONDecodeError as exc:
                raise ValueError("Judge raises controlled JSON error") from exc
            except ValidationError as pexc:
                raise ValueError("Judge raises Pydantic ValidationError") from pexc

        raise ValueError(provider_result.error)
