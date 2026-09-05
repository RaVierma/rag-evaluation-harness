import pytest

from rag_evaluation.evaluation.quality_gate import EvaluationDecisionBuilder
from rag_evaluation.models import EvaluationReport, EvaluationThresholds


@pytest.fixture
def thresholds():
    return EvaluationThresholds(
        min_retrieval_recall=0.92,
        min_retrieval_precision=0.85,
        min_retrieval_mrr=0.80,
        min_context_precision=0.90,
        min_groundedness=0.91,
        min_correctness=0.92,
        max_p95_latency_ms=5000,
        max_avg_cost_usd=0.020,
    )


def test_eval_decision_for_success(thresholds):
    eval_report = EvaluationReport(
        total_cases=5,
        retrieval_recall=0.95,
        retrieval_precision=0.90,
        retrieval_mrr=0.90,
        context_recall=0.91,
        context_precision=0.94,
        groundedness=0.95,
        correctness=0.92,
        relevance=0.90,
        avg_latency_ms=4000,
        p95_latency_ms=2900,
        avg_input_tokens=200,
        avg_output_tokens=400,
        avg_cost_usd=0.015,
        case_results=[],
    )

    eval_decision_builder = EvaluationDecisionBuilder()
    result = eval_decision_builder.build(eval_report, thresholds)

    assert result.is_passed == True
    assert len(result.failures) == 0


def test_eval_decision_for_failed_due_to_recall(thresholds):
    eval_report = EvaluationReport(
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
        p95_latency_ms=2900,
        avg_input_tokens=200,
        avg_output_tokens=400,
        avg_cost_usd=0.015,
        case_results=[],
    )

    eval_decision_builder = EvaluationDecisionBuilder()
    result = eval_decision_builder.build(eval_report, thresholds)

    assert result.is_passed == False
    assert len(result.failures) == 1


def test_eval_decision_for_failed_due_to_p95(thresholds):
    eval_report = EvaluationReport(
        total_cases=5,
        retrieval_recall=0.93,
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
        avg_cost_usd=0.015,
        case_results=[],
    )

    eval_decision_builder = EvaluationDecisionBuilder()
    result = eval_decision_builder.build(eval_report, thresholds)

    assert result.is_passed == False
    assert len(result.failures) == 1


def test_eval_decision_for_exact_boundary(thresholds):
    eval_report = EvaluationReport(
        total_cases=5,
        retrieval_recall=0.92,
        retrieval_precision=0.90,
        retrieval_mrr=0.90,
        context_recall=0.91,
        context_precision=0.94,
        groundedness=0.95,
        correctness=0.92,
        relevance=0.90,
        avg_latency_ms=4000,
        p95_latency_ms=5000,
        avg_input_tokens=200,
        avg_output_tokens=400,
        avg_cost_usd=0.015,
        case_results=[],
    )

    eval_decision_builder = EvaluationDecisionBuilder()
    result = eval_decision_builder.build(eval_report, thresholds)

    assert result.is_passed == True
    assert len(result.failures) == 0


def test_eval_decision_for_multiple_faliure(thresholds):
    eval_report = EvaluationReport(
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

    eval_decision_builder = EvaluationDecisionBuilder()
    result = eval_decision_builder.build(eval_report, thresholds)

    assert result.is_passed == False
    assert len(result.failures) == 3


def test_eval_decision_for_missing_report_metrics(thresholds):

    with pytest.raises(ValueError):
        thresholds.min_input_token = 100  # add this missing report test

        eval_report = EvaluationReport(
            total_cases=5,
            retrieval_recall=0.92,
            retrieval_precision=0.90,
            retrieval_mrr=0.90,
            context_recall=0.91,
            context_precision=0.94,
            groundedness=0.95,
            correctness=0.92,
            relevance=0.90,
            avg_latency_ms=4000,
            p95_latency_ms=5000,
            avg_input_tokens=200,
            avg_output_tokens=400,
            avg_cost_usd=0.015,
            case_results=[],
        )

        eval_decision_builder = EvaluationDecisionBuilder()
        result = eval_decision_builder.build(eval_report, thresholds)

        assert result.is_passed == True
        assert len(result.failures) == 0
