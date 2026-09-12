from enum import Enum

from pydantic import BaseModel

from rag_evaluation.models import (
    EvaluationCaseCategory,
    EvaluationDecision,
    GenerationEvaluation,
    SystemOutput,
)


class RAGDiagnosis(str, Enum):
    PASS = "pass"
    GENERATION_JUDGE_UNCERTAIN = "generation_judge_uncertain"
    RETRIEVAL_FAILURE = "retrieval_failure"
    LUCKY = "lucky"


class EvaluationCaseResult(BaseModel):
    case_id: str
    system_output: SystemOutput

    # Retrieval
    retrieval_recall: float
    retrieval_precision: float
    retrieval_rr: float

    # Context
    context_recall: float
    context_precision: float

    # Generation
    generation: GenerationEvaluation

    diagnosis: RAGDiagnosis


class EvaluationReport(BaseModel):
    total_cases: int

    # Retrieval
    retrieval_recall: float
    retrieval_precision: float
    retrieval_mrr: float

    # Context
    context_recall: float
    context_precision: float

    # Generation
    groundedness: float
    correctness: float
    relevance: float

    # System
    avg_latency_ms: float
    p95_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_cost_usd: float

    case_results: list[EvaluationCaseResult]


class MetricComparison(BaseModel):
    name: str
    v1_value: float
    v2_value: float
    absolute_delta: float
    percentage_delta: float | None


class EvaluationComparison(BaseModel):
    v1_report: EvaluationReport
    v2_report: EvaluationReport
    metric_comparisons: list[MetricComparison]
    v1_decision: EvaluationDecision
    v2_decision: EvaluationDecision


class CategoryEvaluation(BaseModel):
    total_cases: int

    category: EvaluationCaseCategory

    # Retrieval
    retrieval_recall: float
    retrieval_precision: float
    retrieval_mrr: float

    # Context
    context_recall: float
    context_precision: float

    # Generation
    groundedness: float
    correctness: float
    relevance: float

    avg_latency_ms: float
    p95_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_cost_usd: float


class CategoryEvaluationReport(BaseModel):
    total_cases: int
    categories: list[CategoryEvaluation]
