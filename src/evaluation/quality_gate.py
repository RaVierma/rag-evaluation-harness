from evaluation.models.decisions import (
    EvaluationDecision,
    EvaluationThresholds,
    Metric,
    MetricDirection,
)
from evaluation.models.reports import EvaluationReport


class EvaluationDecisionBuilder:
    def build(
        self,
        report: EvaluationReport,
        thresholds: EvaluationThresholds,
    ) -> EvaluationDecision:

        passed = []
        failures = []
        report_data = report.model_dump()

        for name, threshold in thresholds.model_dump().items():
            direction, metric_name = name.split("_", 1)
            actual_score = report_data.get(metric_name)

            if actual_score is None:
                raise ValueError(
                    f"Metric '{metric_name}' not found in evaluation report"
                )

            metric = Metric(
                name=metric_name,
                actual_score=actual_score,
                required_score=threshold,
                direction=MetricDirection.MIN
                if direction == "min"
                else MetricDirection.MAX,
            )

            if direction == "min":
                metric_passed = actual_score >= threshold
            else:
                metric_passed = actual_score <= threshold

            if metric_passed:
                passed.append(metric)
            else:
                failures.append(metric)

        return EvaluationDecision(
            is_passed=not failures, passed=passed, failures=failures
        )
