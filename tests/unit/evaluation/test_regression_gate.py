from evaluation.models.decisions import RegressionThresholds
from evaluation.models.reports import MetricComparison
from evaluation.regression_gate import RegressionDecisionBuilder


def test_regression_for_allowed_limit():
    comparisions = [
        MetricComparison.model_construct(
            name="retrieval_recall",
            v1_value=0.90,
            v2_value=0.88,
            absolute_delta=-0.02,
        ),
        MetricComparison.model_construct(
            name="p95_latency_ms",
            v1_value=4000,
            v2_value=4500,
            absolute_delta=500,
        ),
    ]
    thresholds = RegressionThresholds.model_construct(
        min_retrieval_recall_drop=0.03,
        max_p95_latency_ms_up=1000,
    )

    regression_builder = RegressionDecisionBuilder()

    result = regression_builder.build(comparisions, thresholds)

    assert result.is_passed == True

    assert len(result.passed) == 2


def test_regression_for_exceeds_limit():
    comparisions = [
        MetricComparison.model_construct(
            name="retrieval_recall",
            v1_value=0.90,
            v2_value=0.86,
            absolute_delta=-0.04,
        ),
        MetricComparison.model_construct(
            name="p95_latency_ms",
            v1_value=4000,
            v2_value=4500,
            absolute_delta=500,
        ),
    ]
    thresholds = RegressionThresholds.model_construct(
        min_retrieval_recall_drop=0.03,
        max_p95_latency_ms_up=1000,
    )

    regression_builder = RegressionDecisionBuilder()

    result = regression_builder.build(comparisions, thresholds)

    assert result.is_passed == False

    assert len(result.passed) == 1
    assert len(result.failures) == 1
