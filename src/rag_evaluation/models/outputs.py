from enum import Enum

from pydantic import BaseModel, model_validator

from rag_evaluation.types.strings import NonEmptyString


class SystemOutput(BaseModel):
    retrieved_document_ids: list[str]
    context_document_ids: list[str]
    context: str
    generated_answer: NonEmptyString


class LLMProviderErrorType(str, Enum):
    RATE_LIMIT: str = "RATE_LIMIT"
    TIMEOUT: str = "TIMEOUT"
    SERVICE_UNAVAILABLE: str = "SERVICE_UNAVAILABLE"
    AUTHENTICATION: str = "AUTHENTICATION"
    INVALID_REQUEST: str = "INVALID_REQUEST"
    UNKNOWN: str = "UNKNOWN"


class LLMProviderResult(BaseModel):
    success: bool
    content: str | None = None
    error: str | None = None
    error_type: LLMProviderErrorType | None = None
    raw_response: object | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    @model_validator(mode="after")
    def validate_model(self):
        if (self.success or self.content) and self.error:
            raise ValueError("success request must contain content but no error")

        if not self.success and (not self.error or not self.error_type):
            raise ValueError("fail request must contain error and error_type")

        return self
