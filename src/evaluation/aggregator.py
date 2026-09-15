from evaluation.metrics.performance import calculate_p95
from evaluation.models.reports import EvaluationCaseResult


class EvaluationMetricsAggregator:
    def aggregate(
        self,
        eval_case_results: list[EvaluationCaseResult],
    ) -> dict:
        total_cases = len(eval_case_results)
        return {
            "retrieval_recall": sum(
                eval_case.retrieval_recall for eval_case in eval_case_results
            )
            / total_cases,
            "retrieval_precision": sum(
                eval_case.retrieval_precision for eval_case in eval_case_results
            )
            / total_cases,
            "retrieval_mrr": sum(
                eval_case.retrieval_rr for eval_case in eval_case_results
            )
            / total_cases,
            "context_recall": sum(
                eval_case.context_recall for eval_case in eval_case_results
            )
            / total_cases,
            "context_precision": sum(
                eval_case.context_precision for eval_case in eval_case_results
            )
            / total_cases,
            "groundedness": sum(
                eval_case.generation.groundedness.score
                for eval_case in eval_case_results
            )
            / total_cases,
            "correctness": sum(
                eval_case.generation.correctness.score
                for eval_case in eval_case_results
            )
            / total_cases,
            "relevance": sum(
                eval_case.generation.relevance.score for eval_case in eval_case_results
            )
            / total_cases,
            "avg_latency_ms": sum(
                eval_case.performance.total_latency_ms
                for eval_case in eval_case_results
            )
            / total_cases,
            "p95_latency_ms": calculate_p95(
                [
                    eval_case.performance.total_latency_ms
                    for eval_case in eval_case_results
                ]
            ),
            "avg_input_tokens": sum(
                eval_case.performance.input_tokens for eval_case in eval_case_results
            )
            / total_cases,
            "avg_output_tokens": sum(
                eval_case.performance.output_tokens for eval_case in eval_case_results
            )
            / total_cases,
            "avg_cost_usd": sum(
                eval_case.performance.cost_usd for eval_case in eval_case_results
            )
            / total_cases,
        }
