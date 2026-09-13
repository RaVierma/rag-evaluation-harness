from pydantic import BaseModel, Field

from rag_evaluation.models.chunks import RerankedChunk
from rag_evaluation.models.outputs import SystemOutput
from rag_evaluation.models.performance import RAGPerformance


class RetrievalOutput(BaseModel):
    query: str
    results: list[RerankedChunk]
    candidate_k: int = Field(gt=0)
    top_k: int = Field(gt=0)


class RAGPipelineResult(BaseModel):
    output: SystemOutput
    performance: RAGPerformance
