import json

from pydantic import BaseModel

from rag.models.outputs import LLMProviderErrorType, LLMProviderResult
from rag.providers.generation.base import LLMProvider


class DummyLLMProvider(LLMProvider):
    def __init__(self, response: str | None = None):
        self.response = response

    def call(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int = -1,
    ) -> LLMProviderResult:
        response = (
            json.dumps(
                {
                    "groundedness": {
                        "score": 1.0,
                        "reason": "All claims are supported.",
                        "evidence": [
                            {
                                "claim": "The allowance is $500.",
                                "support": "The allowance is $500.",
                            }
                        ],
                    },
                    "correctness": {
                        "score": 1.0,
                        "reason": "The answer matches the expected answer.",
                        "evidence": [
                            {
                                "claim": "The allowance is $500.",
                                "support": "The expected answer states that the allowance is $500.",
                            }
                        ],
                    },
                    "relevance": {
                        "score": 1.0,
                        "reason": "The answer directly addresses the question.",
                        "evidence": [
                            {
                                "claim": "The allowance is $500.",
                                "support": "The answer directly addresses the allowance question.",
                            }
                        ],
                    },
                }
            )
            if self.response is None
            else self.response
        )

        success = True
        content = None
        error = None
        error_type = None

        if prompt:
            content = response
        else:
            success = False
            error = "not valid prompt"
            error_type = LLMProviderErrorType.INVALID_REQUEST

        return LLMProviderResult(
            success=success, content=content, error=error, error_type=error_type
        )
