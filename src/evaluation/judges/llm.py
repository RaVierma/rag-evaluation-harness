import json

from pydantic_core import ValidationError

from evaluation.judges.base import Judge
from evaluation.models.generation import GenerationEvaluation
from evaluation.prompts.loader import load_eval_prompt
from rag.providers.generation.base import LLMProvider


class LLMJudge(Judge):
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

        prompt = load_eval_prompt(
            "v1", question, context, expected_answer, generated_answer
        )

        provider_result = self.provider.call(
            prompt, response_schema=GenerationEvaluation
        )

        if provider_result.success:
            try:
                content = provider_result.content
                evaluation_data = json.loads(content)
                return GenerationEvaluation(**evaluation_data)
            except json.JSONDecodeError as exc:
                print("RAW LLM RESPONSE:")
                print(provider_result.content)
                print("JSON ERROR:")
                print(exc)
                raise ValueError("Judge raises controlled JSON error") from exc
            except ValidationError as pexc:
                print("RAW LLM RESPONSE:")
                print(provider_result.content)
                print("PYDANTIC ERROR:")
                print(pexc)
                raise ValueError("Judge raises Pydantic ValidationError") from pexc

        raise ValueError(provider_result.error)
