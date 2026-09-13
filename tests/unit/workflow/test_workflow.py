from unittest.mock import Mock

import pytest

from rag_evaluation.evaluation.workflow import EvaluationWorkflow
from rag_evaluation.models import (
    EvaluationCase,
    EvaluationCaseCategory,
    EvaluationComparison,
    EvaluationDecision,
    EvaluationThresholds,
    GenerationEvaluation,
    MetricDirection,
    RelevanceJudgment,
    RegressionDecision,
    RegressionThresholds,
    SystemOutput,
)
from rag_evaluation.models.generation import MetricEvaluation
from rag_evaluation.models.performance import RAGPerformance
from rag_evaluation.models.reports import (
    EvaluationCaseResult,
    RAGDiagnosis,
)
from rag_evaluation.models.pipeline import RAGPipelineResult


def _make_case(case_id: str = "case-001") -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        question="What is the leave policy?",
        expected_answer="Employees receive 20 days of annual leave.",
        relevant_document_ids=["doc-001"],
        relevance_judgments=[
            RelevanceJudgment(
                chunk_id="chunk-001",
                relevance=3,
            )
        ],
        category=EvaluationCaseCategory.SIMPLE,
    )


def _make_pipeline_result(
    latency_ms: float = 100.0,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cost_usd: float = 0.01,
) -> RAGPipelineResult:
    output = SystemOutput(
        retrieved_document_ids=["doc-001"],
        context_document_ids=["doc-001"],
        context="Employees receive 20 days of annual leave.",
        generated_answer="Employees receive 20 days of annual leave.",
    )

    performance = RAGPerformance(
        total_latency_ms=latency_ms,
        retrieval_latency_ms=10.0,
        context_latency_ms=1.0,
        llm_latency_ms=89.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )

    return RAGPipelineResult(
        output=output,
        performance=performance,
    )


def _make_case_result(
    case_id: str = "case-001",
    latency_ms: float = 100.0,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cost_usd: float = 0.01,
) -> EvaluationCaseResult:
    generation = GenerationEvaluation(
        groundedness=MetricEvaluation(
            score=1.0,
            reason="The answer is fully supported by the context.",
            evidence=[],
        ),
        correctness=MetricEvaluation(
            score=1.0,
            reason="The answer matches the expected answer.",
            evidence=[],
        ),
        relevance=MetricEvaluation(
            score=1.0,
            reason="The answer directly addresses the question.",
            evidence=[],
        ),
    )

    return EvaluationCaseResult(
        case_id=case_id,
        system_output=_make_pipeline_result().output,
        retrieval_recall=1.0,
        retrieval_precision=0.8,
        retrieval_rr=1.0,
        context_recall=1.0,
        context_precision=0.75,
        generation=generation,
        performance=RAGPerformance(
            total_latency_ms=latency_ms,
            retrieval_latency_ms=10.0,
            context_latency_ms=1.0,
            llm_latency_ms=89.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ),
        diagnosis=RAGDiagnosis.PASS,
    )


@pytest.fixture
def workflow():
    evaluation_runner = Mock()
    decision_builder = Mock()
    comparison_builder = Mock()
    regression_builder = Mock()

    return EvaluationWorkflow(
        evaluation_runner=evaluation_runner,
        decision_builder=decision_builder,
        comparison_builder=comparison_builder,
        regression_builder=regression_builder,
    )


def test_evaluate_rejects_empty_cases(workflow):
    with pytest.raises(
        ValueError,
        match="evaluation cases cannot be empty",
    ):
        workflow.evaluate(
            cases=[],
            pipeline_results=[],
        )


def test_evaluate_rejects_case_result_count_mismatch(workflow):
    cases = [_make_case()]

    with pytest.raises(
        ValueError,
        match="number of evaluation cases must match number of pipeline results",
    ):
        workflow.evaluate(
            cases=cases,
            pipeline_results=[],
        )


def test_evaluate_delegates_each_case_to_evaluation_runner(workflow):
    case = _make_case()
    pipeline_result = _make_pipeline_result()

    expected_result = _make_case_result()

    workflow.evaluation_runner.evaluate.return_value = expected_result

    report = workflow.evaluate(
        cases=[case],
        pipeline_results=[pipeline_result],
    )

    workflow.evaluation_runner.evaluate.assert_called_once_with(
        case=case,
        system_output=pipeline_result.output,
        rag_performance=pipeline_result.performance,
    )

    assert report.total_cases == 1
    assert report.case_results == [expected_result]


def test_evaluate_aggregates_case_results(workflow):
    cases = [
        _make_case("case-001"),
        _make_case("case-002"),
    ]

    pipeline_results = [
        _make_pipeline_result(
            latency_ms=100.0,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.01,
        ),
        _make_pipeline_result(
            latency_ms=200.0,
            input_tokens=200,
            output_tokens=40,
            cost_usd=0.03,
        ),
    ]

    workflow.evaluation_runner.evaluate.side_effect = [
        _make_case_result(
            case_id="case-001",
            latency_ms=100.0,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.01,
        ),
        _make_case_result(
            case_id="case-002",
            latency_ms=200.0,
            input_tokens=200,
            output_tokens=40,
            cost_usd=0.03,
        ),
    ]

    report = workflow.evaluate(
        cases=cases,
        pipeline_results=pipeline_results,
    )

    assert report.total_cases == 2

    assert report.retrieval_recall == pytest.approx(1.0)
    assert report.retrieval_precision == pytest.approx(0.8)
    assert report.retrieval_mrr == pytest.approx(1.0)

    assert report.context_recall == pytest.approx(1.0)
    assert report.context_precision == pytest.approx(0.75)

    assert report.groundedness == pytest.approx(1.0)
    assert report.correctness == pytest.approx(1.0)
    assert report.relevance == pytest.approx(1.0)

    assert report.avg_latency_ms == pytest.approx(150.0)
    assert report.avg_input_tokens == pytest.approx(150.0)
    assert report.avg_output_tokens == pytest.approx(30.0)
    assert report.avg_cost_usd == pytest.approx(0.02)


def test_evaluate_calculates_p95_latency(workflow):
    latencies = [
        100.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
        1000.0,
    ]

    cases = [_make_case(f"case-{index}") for index in range(10)]

    pipeline_results = [
        _make_pipeline_result(latency_ms=latency) for latency in latencies
    ]

    workflow.evaluation_runner.evaluate.side_effect = [
        _make_case_result(
            case_id=f"case-{index}",
            latency_ms=latency,
        )
        for index, latency in enumerate(latencies)
    ]

    report = workflow.evaluate(
        cases=cases,
        pipeline_results=pipeline_results,
    )

    assert report.p95_latency_ms == pytest.approx(955.0)


def test_quality_gate_delegates_to_decision_builder(workflow):
    report = Mock()
    thresholds = EvaluationThresholds(
        min_retrieval_recall=0.9,
        min_retrieval_precision=0.7,
        min_retrieval_mrr=0.9,
        min_context_precision=0.7,
        min_groundedness=0.9,
        min_correctness=0.9,
        max_p95_latency_ms=5000.0,
        max_avg_cost_usd=1.0,
    )

    expected_decision = EvaluationDecision(
        is_passed=True,
        passed=[],
        failures=[],
    )

    workflow.decision_builder.build.return_value = expected_decision

    result = workflow.quality_gate(
        report=report,
        thresholds=thresholds,
    )

    workflow.decision_builder.build.assert_called_once_with(
        report=report,
        thresholds=thresholds,
    )

    assert result is expected_decision


def test_compare_delegates_to_comparison_builder(workflow):
    v1_report = Mock()
    v2_report = Mock()

    v1_decision = EvaluationDecision(
        is_passed=True,
        passed=[],
        failures=[],
    )

    v2_decision = EvaluationDecision(
        is_passed=True,
        passed=[],
        failures=[],
    )

    expected_comparison = Mock(spec=EvaluationComparison)

    workflow.comparison_builder.build.return_value = expected_comparison

    result = workflow.compare(
        v1_report=v1_report,
        v2_report=v2_report,
        v1_decision=v1_decision,
        v2_decision=v2_decision,
    )

    workflow.comparison_builder.build.assert_called_once_with(
        v1_report=v1_report,
        v2_report=v2_report,
        v1_decision=v1_decision,
        v2_decision=v2_decision,
    )

    assert result is expected_comparison


def test_regression_gate_delegates_to_regression_builder(workflow):
    comparison = Mock()
    thresholds = RegressionThresholds(
        min_retrieval_recall_drop=0.03,
        min_retrieval_precision_drop=0.03,
        min_context_recall_drop=0.03,
        min_context_precision_drop=0.03,
        min_groundedness_drop=0.03,
        min_correctness_drop=0.03,
        min_relevance_drop=0.03,
        max_p95_latency_ms_up=0.03,
        max_avg_cost_usd_up=0.02,
    )

    expected_decision = RegressionDecision(
        is_passed=True,
        passed=[],
        failures=[],
    )

    workflow.regression_builder.build.return_value = expected_decision

    comparison.metric_comparisons = []

    result = workflow.regression_gate(
        comparison=comparison,
        thresholds=thresholds,
    )

    workflow.regression_builder.build.assert_called_once_with(
        metric_comparisons=comparison.metric_comparisons,
        thresholds=thresholds,
    )

    assert result is expected_decision
