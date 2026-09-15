from collections.abc import Sequence

from tqdm import tqdm

from evaluation.comparison import EvaluationComparisonBuilder
from evaluation.models.cases import EvaluationCase
from evaluation.models.decisions import (
    EvaluationDecision,
    EvaluationThresholds,
    RegressionDecision,
    RegressionThresholds,
)
from evaluation.models.reports import EvaluationComparison, EvaluationReport
from evaluation.quality_gate import EvaluationDecisionBuilder
from evaluation.regression_gate import RegressionDecisionBuilder
from evaluation.runner import EvaluationRunner
from rag.models.pipeline import RAGPipelineResult


class EvaluationWorkflow:
    """Orchestrates the end-to-end RAG evaluation workflow."""

    def __init__(
        self,
        evaluation_runner: EvaluationRunner,
        decision_builder: EvaluationDecisionBuilder,
        comparison_builder: EvaluationComparisonBuilder,
        regression_builder: RegressionDecisionBuilder,
    ) -> None:
        self.evaluation_runner = evaluation_runner
        self.decision_builder = decision_builder
        self.comparison_builder = comparison_builder
        self.regression_builder = regression_builder

    def evaluate(
        self,
        cases: Sequence[EvaluationCase],
        pipeline_results: Sequence[RAGPipelineResult],
    ) -> EvaluationReport:
        """Evaluate all cases and aggregate the results into a report."""

        if not cases:
            raise ValueError("evaluation cases cannot be empty")

        if len(cases) != len(pipeline_results):
            raise ValueError(
                "number of evaluation cases must match number of pipeline results"
            )

        case_results = [
            self.evaluation_runner.evaluate(
                case=case,
                system_output=pipeline_result.output,
                rag_performance=pipeline_result.performance,
            )
            for case, pipeline_result in tqdm(
                zip(
                    cases,
                    pipeline_results,
                    strict=True,
                ),
                desc="Evaluating Cases",
                total=len(cases),
            )
        ]

        return self._build_report(case_results)

    def quality_gate(
        self,
        report: EvaluationReport,
        thresholds: EvaluationThresholds,
    ) -> EvaluationDecision:
        """Apply the quality gate to an evaluation report."""

        return self.decision_builder.build(
            report=report,
            thresholds=thresholds,
        )

    def compare(
        self,
        v1_report: EvaluationReport,
        v2_report: EvaluationReport,
        v1_decision: EvaluationDecision,
        v2_decision: EvaluationDecision,
    ) -> EvaluationComparison:
        """Compare two evaluation runs."""

        return self.comparison_builder.build(
            v1_report=v1_report,
            v2_report=v2_report,
            v1_decision=v1_decision,
            v2_decision=v2_decision,
        )

    def regression_gate(
        self,
        comparison: EvaluationComparison,
        thresholds: RegressionThresholds,
    ) -> RegressionDecision:
        """Apply the regression gate to a V1/V2 comparison."""

        return self.regression_builder.build(
            metric_comparisons=comparison.metric_comparisons,
            thresholds=thresholds,
        )

    @staticmethod
    def _build_report(
        case_results,
    ) -> EvaluationReport:
        """Aggregate case-level results into an evaluation report."""

        total_cases = len(case_results)

        retrieval_recalls = [result.retrieval_recall for result in case_results]

        retrieval_precisions = [result.retrieval_precision for result in case_results]

        retrieval_rrs = [result.retrieval_rr for result in case_results]

        context_recalls = [result.context_recall for result in case_results]

        context_precisions = [result.context_precision for result in case_results]

        groundedness_scores = [
            result.generation.groundedness.score for result in case_results
        ]

        correctness_scores = [
            result.generation.correctness.score for result in case_results
        ]

        relevance_scores = [
            result.generation.relevance.score for result in case_results
        ]

        latencies = [result.performance.total_latency_ms for result in case_results]

        input_tokens = [result.performance.input_tokens or 0 for result in case_results]

        output_tokens = [
            result.performance.output_tokens or 0 for result in case_results
        ]

        costs = [result.performance.cost_usd or 0 for result in case_results]

        return EvaluationReport(
            total_cases=total_cases,
            retrieval_recall=_average(retrieval_recalls),
            retrieval_precision=_average(retrieval_precisions),
            retrieval_mrr=_average(retrieval_rrs),
            context_recall=_average(context_recalls),
            context_precision=_average(context_precisions),
            groundedness=_average(groundedness_scores),
            correctness=_average(correctness_scores),
            relevance=_average(relevance_scores),
            avg_latency_ms=_average(latencies),
            p95_latency_ms=_percentile(latencies, 95),
            avg_input_tokens=_average(input_tokens),
            avg_output_tokens=_average(output_tokens),
            avg_cost_usd=_average(costs),
            case_results=case_results,
        )


def _average(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate average of empty values")

    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty values")

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile / 100

    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)

    weight = position - lower

    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight
