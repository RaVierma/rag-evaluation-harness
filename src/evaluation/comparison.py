from evaluation.constants import COMPARABLE_METRICS
from evaluation.models.decisions import EvaluationDecision
from evaluation.models.reports import (
    EvaluationComparison,
    EvaluationReport,
    MetricComparison,
)


class EvaluationComparisonBuilder:
    def build(
        self,
        v1_report: EvaluationReport,
        v2_report: EvaluationReport,
        v1_decision: EvaluationDecision,
        v2_decision: EvaluationDecision,
    ) -> EvaluationComparison:

        v1_report_data = v1_report.model_dump()
        v2_report_data = v2_report.model_dump()

        metric_comparisons = []

        for metric_name in COMPARABLE_METRICS:
            value = v1_report_data.get(metric_name)
            if value is None:
                raise ValueError(
                    f"Metric '{metric_name}' not found in v1 evaluation report"
                )

            v2_value = v2_report_data.get(metric_name)

            if v2_value is None:
                raise ValueError(
                    f"Metric '{metric_name}' not found in v2 evaluation report"
                )

            absolute_delta = v2_value - value

            percentage_delta = self._calculate_percentage_delta(value, v2_value)

            metric_comparision = MetricComparison(
                name=metric_name,
                v1_value=value,
                v2_value=v2_value,
                absolute_delta=absolute_delta,
                percentage_delta=percentage_delta,
            )

            metric_comparisons.append(metric_comparision)

        return EvaluationComparison(
            v1_report=v1_report,
            v2_report=v2_report,
            metric_comparisons=metric_comparisons,
            v1_decision=v1_decision,
            v2_decision=v2_decision,
        )

    def _calculate_percentage_delta(self, v1, v2):
        if v1 == 0:
            return None
        else:
            return ((v2 - v1) / v1) * 100
