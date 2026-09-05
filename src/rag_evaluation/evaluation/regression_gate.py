from rag_evaluation.models import (
    MetricComparison,
    MetricDirection,
    RegressionDecision,
    RegressionMetric,
    RegressionThresholds,
)


class RegressionDecisionBuilder:
    def build(
        self,
        metric_comparisons: list[MetricComparison],
        thresholds: RegressionThresholds,
    ) -> RegressionDecision:

        passed = []
        failures = []
        metric_data = {metric.name: metric for metric in metric_comparisons}

        for name, threshold in thresholds.model_dump().items():
            direction, metric_name = name.split("_", 1)
            metric_name = metric_name.rsplit("_", 1)[0]
            metric = metric_data.get(metric_name)

            if metric is None:
                raise ValueError(
                    f"Metric '{metric_name}' not found in metric comparisons"
                )

            actual_delta = metric.absolute_delta

            metric = RegressionMetric(
                name=metric_name,
                v1_value=metric.v1_value,
                v2_value=metric.v2_value,
                actual_delta=actual_delta,
                threshold_delta=threshold,
                direction=MetricDirection.MIN
                if direction == "min"
                else MetricDirection.MAX,
            )

            if direction == "min":
                metric_passed = actual_delta >= -threshold
            else:
                metric_passed = actual_delta <= threshold

            if metric_passed:
                passed.append(metric)
            else:
                failures.append(metric)

        return RegressionDecision(
            is_passed=not failures, passed=passed, failures=failures
        )
