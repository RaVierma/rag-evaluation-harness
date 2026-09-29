from pydantic import BaseModel

from evaluation.models.reports import EvaluationReport


class EvaluationMetrics(BaseModel):
    retrieval_recall: float
    retrieval_precision: float
    retrieval_mrr: float

    context_recall: float
    context_precision: float

    groundedness: float
    correctness: float
    relevance: float

    avg_latency_ms: float
    p95_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_cost_usd: float


class EvaluationArtifact(BaseModel):
    version: str
    dataset: str
    total_cases: int
    metrics: EvaluationMetrics

    @classmethod
    def from_report(
        cls,
        *,
        version: str,
        dataset: str,
        report: EvaluationReport,
    ) -> "EvaluationArtifact":
        return cls(
            version=version,
            dataset=dataset,
            total_cases=report.total_cases,
            metrics=EvaluationMetrics(
                retrieval_recall=report.retrieval_recall,
                retrieval_precision=report.retrieval_precision,
                retrieval_mrr=report.retrieval_mrr,
                context_recall=report.context_recall,
                context_precision=report.context_precision,
                groundedness=report.groundedness,
                correctness=report.correctness,
                relevance=report.relevance,
                avg_latency_ms=report.avg_latency_ms,
                p95_latency_ms=report.p95_latency_ms,
                avg_input_tokens=report.avg_input_tokens,
                avg_output_tokens=report.avg_output_tokens,
                avg_cost_usd=report.avg_cost_usd,
            ),
        )
