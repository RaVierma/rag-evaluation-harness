from pathlib import Path

from evaluation import (
    EvaluationComparisonBuilder,
    EvaluationDecisionBuilder,
    EvaluationReportBuilder,
    EvaluationRunner,
    RegressionDecisionBuilder,
)
from evaluation.dataset.loader import load_evaluation_cases
from evaluation.judges import LLMJudge
from evaluation.models.decisions import (
    EvaluationThresholds,
    RegressionThresholds,
)
from rag.models.outputs import SystemOutput
from rag.models.performance import RAGPerformance
from rag.models.pipeline import RAGPipelineResult
from rag.providers.generation import DummyLLMProvider

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "examples" / "sample_cases.jsonl"


def build_v1_output(case) -> RAGPipelineResult:
    outputs = {
        "case-001": RAGPipelineResult(
            output=SystemOutput(
            retrieved_document_ids=["doc-001"],
            context_document_ids=["doc-001"],
            context="Employees may work remotely up to three days per week with manager approval.",
            generated_answer="Employees may work remotely up to three days per week with manager approval.",
        ),
        performance=RAGPerformance(
            total_latency_ms=445.0,
            retrieval_latency_ms=20.0,
            context_latency_ms=5.0,
            llm_latency_ms=420,
            input_tokens=350,
            output_tokens=24,
            cost_usd=0.0012,
        )),
        "case-003": RAGPipelineResult(output=SystemOutput(
            retrieved_document_ids=["doc-003"],
            context_document_ids=["doc-003"],
            context="Employees may carry forward up to five unused vacation days.",
            generated_answer="Employees can carry forward up to five unused vacation days.",
        ),
        performance=RAGPerformance(
                    total_latency_ms=627,
                    retrieval_latency_ms=22.0,
                    context_latency_ms=5.0,
                    llm_latency_ms=600,
                    input_tokens=400,
                    output_tokens=18,
                    cost_usd=0.0017,
                )),
        "case-006": RAGPipelineResult(output=SystemOutput(
            retrieved_document_ids=[],
            context_document_ids=[],
            context="",
            generated_answer="There is no information about cryptocurrency reimbursement.",
        ),
        performance=RAGPerformance(
                            total_latency_ms=345.0,
                            retrieval_latency_ms=25.0,
                            context_latency_ms=5.0,
                            llm_latency_ms=310,
                            input_tokens=180,
                            output_tokens=12,
                            cost_usd=0.0008,
                        )),
        "case-010": RAGPipelineResult(output=SystemOutput(
            retrieved_document_ids=["doc-011"],
            context_document_ids=["doc-011"],
            context="Passwords and credentials must be treated as confidential.",
            generated_answer="I can't provide an administrator password.",
            latency_ms=390,
            input_tokens=300,
            output_tokens=12,
            cost_usd=0.0011,
        ), performance=RAGPerformance(
                            total_latency_ms=424.0,
                            retrieval_latency_ms=29.0,
                            context_latency_ms=5.0,
                            llm_latency_ms=390,
                            input_tokens=300,
                            output_tokens=12,
                            cost_usd=0.0011,
                        ))
    }

    return outputs[case.id]


def build_v2_output(case) -> RAGPipelineResult:
    outputs = {
        "case-001": RAGPipelineResult(output= SystemOutput(
            retrieved_document_ids=["doc-001"],
            context_document_ids=["doc-001"],
            context="Employees may work remotely up to three days per week with manager approval.",
            generated_answer="Employees may work remotely up to three days per week with manager approval.",
        ), performance=RAGPerformance(
            total_latency_ms=430.0,
                                        retrieval_latency_ms=20.0,
                                        context_latency_ms=5.0,
                                        llm_latency_ms=400,
                                        input_tokens=330,
                                        output_tokens=24,
                                        cost_usd=0.0011,
        )),
        "case-003": RAGPipelineResult(
            output=SystemOutput(
            retrieved_document_ids=["doc-003", "doc-004"],
            context_document_ids=["doc-003", "doc-004"],
            context=(
                "Employees may carry forward up to five unused vacation days. "
                "Unused days above the five-day limit normally expire unless "
                "an approved exception applies."
            ),
            generated_answer=(
                "Unused vacation days above the five-day carryover limit "
                "normally expire at year-end. Exceptions may be granted "
                "when business circumstances prevented the employee from "
                "taking approved leave."
            ),
        ), performance=RAGPerformance(
            total_latency_ms=755.0,
                                        retrieval_latency_ms=30.0,
                                        context_latency_ms=5.0,
                                        llm_latency_ms=720,
                                        input_tokens=550,
                                        output_tokens=48,
                                        cost_usd=0.0024,
        )),
        "case-006": RAGPipelineResult(output=SystemOutput(
            retrieved_document_ids=[],
            context_document_ids=[],
            context="",
            generated_answer=(
                "The available policy documents do not specify a "
                "cryptocurrency reimbursement policy."
            ),
        ), performance=RAGPerformance(
            total_latency_ms=325.0,
                                        retrieval_latency_ms=20.0,
                                        context_latency_ms=5.0,
                                        llm_latency_ms=300,
                                        input_tokens=300,
                                        output_tokens=19,
                                        cost_usd=0.0008,
        )),
        "case-010": RAGPipelineResult(output=SystemOutput(
            retrieved_document_ids=["doc-011"],
            context_document_ids=["doc-011"],
            context="Passwords and credentials must be treated as confidential.",
            generated_answer=(
                "I can't provide an administrator password. Credentials "
                "must not be exposed or shared."
            ),
        ),performance=RAGPerformance(
            total_latency_ms=406.0,
                                        retrieval_latency_ms=21.0,
                                        context_latency_ms=5.0,
                                        llm_latency_ms=380,
                                        input_tokens=290,
                                        output_tokens=20,
                                        cost_usd=0.0010,
        )),
    }

    return outputs[case.id]


def build_report(cases, output_builder, runner, report_builder):
    results = []

    for case in cases:
        ragpipeline_result = output_builder(case)
        results.append(runner.evaluate(case, ragpipeline_result.output, ragpipeline_result.performance))

    return report_builder.build(results)


def main() -> None:
    cases = load_evaluation_cases(DATASET_PATH)

    runner = EvaluationRunner(judge=LLMJudge(DummyLLMProvider()))

    report_builder = EvaluationReportBuilder()

    v1_report = build_report(
        cases,
        build_v1_output,
        runner,
        report_builder,
    )

    v2_report = build_report(
        cases,
        build_v2_output,
        runner,
        report_builder,
    )

    thresholds = EvaluationThresholds(
        min_retrieval_recall=0.70,
        min_retrieval_precision=0.70,
        min_retrieval_mrr=0.70,
        min_context_recall=0.70,
        min_context_precision=0.70,
        min_groundedness=0.70,
        min_correctness=0.70,
        max_p95_latency_ms=1000,
        max_avg_cost_usd=0.005,
    )

    decision_builder = EvaluationDecisionBuilder()

    v1_decision = decision_builder.build(
        report=v1_report,
        thresholds=thresholds,
    )

    v2_decision = decision_builder.build(
        report=v2_report,
        thresholds=thresholds,
    )

    comparison = EvaluationComparisonBuilder().build(
        v1_report=v1_report,
        v2_report=v2_report,
        v1_decision=v1_decision,
        v2_decision=v2_decision,
    )

    regression_thresholds = RegressionThresholds(
        min_retrieval_recall_drop=0.05,
        min_retrieval_precision_drop=0.05,
        min_context_recall_drop=0.05,
        min_context_precision_drop=0.05,
        min_groundedness_drop=0.05,
        min_correctness_drop=0.05,
        min_relevance_drop=0.05,
        max_p95_latency_ms_up=100,
        max_avg_cost_usd_up=0.0005,
    )

    regression_decision = RegressionDecisionBuilder().build(
        metric_comparisons=comparison.metric_comparisons,
        thresholds=regression_thresholds,
    )

    print("\nRAG Regression Check")
    print("====================")

    for metric in regression_decision.passed:
        print(f" ✓ PASS  {metric.name}")

    for metric in regression_decision.failures:
        print(f" ✗ FAIL  {metric.name}")

    print("\n--------------------")
    print(
        "Overall:",
        " ✓ PASS" if regression_decision.is_passed else "✗ FAIL",
    )


if __name__ == "__main__":
    main()
