from ollama import Client
from pydantic import BaseModel

from rag_evaluation.models.outputs import LLMProviderErrorType, LLMProviderResult
from rag_evaluation.providers.generation.base import LLMProvider


class OllamaLLMProvider(LLMProvider):
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._client = Client()

    @property
    def model_name(self) -> str:
        return self._model_name

    def call(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMProviderResult:
        if not prompt.strip():
            return LLMProviderResult(
                success=False,
                error="not valid prompt",
                error_type=LLMProviderErrorType.INVALID_REQUEST,
            )
        try:
            generate_kwargs = {
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
                "keep_alive": -1,
            }

            if response_schema is not None:
                generate_kwargs["format"] = response_schema.model_json_schema()

            response = self._client.generate(self.model_name, **generate_kwargs)

            return LLMProviderResult(
                success=True,
                content=response.response,
                error=None,
                error_type=None,
                input_tokens=response.prompt_eval_count,
                output_tokens=response.eval_count,
                cost_usd=(response.eval_count + response.prompt_eval_count) * 0.0015,
            )
        except Exception as e:
            return LLMProviderResult(
                success=False,
                content=None,
                error=str(e),
                error_type=LLMProviderErrorType.UNKNOWN,
            )
