from pydantic import BaseModel, Field

from rag.models.chunks import RerankedChunk
from rag.models.outputs import SystemOutput
from rag.models.performance import RAGPerformance


class RetrievalOutput(BaseModel):
    query: str
    results: list[RerankedChunk]
    candidate_k: int = Field(gt=0)
    top_k: int = Field(gt=0)


class RAGPipelineResult(BaseModel):
    output: SystemOutput
    performance: RAGPerformance
