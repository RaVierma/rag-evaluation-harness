from pydantic import BaseModel, Field


class RAGPerformance(BaseModel):
    total_latency_ms: float = Field(ge=0)
    retrieval_latency_ms: float = Field(ge=0)
    context_latency_ms: float = Field(ge=0)
    llm_latency_ms: float = Field(ge=0)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    generation_tokens_per_second: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
