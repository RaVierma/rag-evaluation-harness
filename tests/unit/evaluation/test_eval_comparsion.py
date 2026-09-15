from evaluation.comparison import EvaluationComparisonBuilder
from evaluation.models.decisions import EvaluationDecision
from evaluation.models.reports import EvaluationReport


def test_eval_comparision_for_all_metrics_compared():
    v1_report = EvaluationReport(
        total_cases=5,
        retrieval_recall=0.88,
        retrieval_precision=0.90,
        retrieval_mrr=0.90,
        context_recall=0.91,
        context_precision=0.94,
        groundedness=0.95,
        correctness=0.92,
        relevance=0.90,
        avg_latency_ms=4000,
        p95_latency_ms=5500,
        avg_input_tokens=200,
        avg_output_tokens=400,
        avg_cost_usd=0.025,
        case_results=[],
    )
    v2_report = EvaluationReport(
        total_cases=5,
        retrieval_recall=0.97,
        retrieval_precision=0.92,
        retrieval_mrr=0.91,
        context_recall=0.94,
        context_precision=0.89,
        groundedness=0.955,
        correctness=0.86,
        relevance=0.0,
        avg_latency_ms=4000,
        p95_latency_ms=6000,
        avg_input_tokens=200,
        avg_output_tokens=400,
        avg_cost_usd=0.020,
        case_results=[],
    )

    v1_decision = EvaluationDecision(is_passed=True, passed=[], failures=[])
    v2_decision = EvaluationDecision(is_passed=True, passed=[], failures=[])

    comparison_builder = EvaluationComparisonBuilder()

    result = comparison_builder.build(
        v1_report=v1_report,
        v2_report=v2_report,
        v1_decision=v1_decision,
        v2_decision=v2_decision,
    )

    for metric_comp in result.metric_comparisons:
        if metric_comp.name == "retrieval_recall":
            assert (
                metric_comp.percentage_delta
                == comparison_builder._calculate_percentage_delta(0.88, 0.97)
            )

        if metric_comp.name == "context_precision":
            assert (
                metric_comp.percentage_delta
                == comparison_builder._calculate_percentage_delta(0.94, 0.89)
            )

        if metric_comp.name == "relevance":
            assert (
                metric_comp.percentage_delta
                == comparison_builder._calculate_percentage_delta(0.90, 0.0)
            )

        if metric_comp.name == "avg_latency_ms":
            assert metric_comp.absolute_delta == 0
            assert (
                metric_comp.percentage_delta
                == comparison_builder._calculate_percentage_delta(4000, 4000)
            )
