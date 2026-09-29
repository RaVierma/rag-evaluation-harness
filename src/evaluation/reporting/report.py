from evaluation.aggregator import EvaluationMetricsAggregator
from evaluation.models.reports import EvaluationCaseResult, EvaluationReport


class EvaluationReportBuilder:
    def __init__(self):
        self.aggregator = EvaluationMetricsAggregator()

    def build(self, eval_case_results: list[EvaluationCaseResult]) -> EvaluationReport:

        if not eval_case_results:
            raise ValueError("evaluation results cannot be empty")

        total_cases = len(eval_case_results)

        metrics = self.aggregator.aggregate(eval_case_results)

        return EvaluationReport(
            total_cases=total_cases,
            **metrics,
            case_results=eval_case_results,
        )
