from pathlib import Path

from rag_evaluation.dataset import load_evaluation_cases
from rag_evaluation.evaluation import (
    EvaluationComparisonBuilder,
    EvaluationDecisionBuilder,
    EvaluationReportBuilder,
    EvaluationRunner,
)
from rag_evaluation.judges.dummy import DummyJudge
from rag_evaluation.models import EvaluationThresholds, SystemOutput
from rag_evaluation.providers.dummy import DummyLLMProvider

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "examples" / "sample_cases.jsonl"


def build_v1_output(case) -> SystemOutput:
    outputs = {
        "case-001": SystemOutput(
            retrieved_document_ids=["doc-001"],
            context_document_ids=["doc-001"],
            context="Employees may work remotely up to three days per week with manager approval.",
            generated_answer="Employees may work remotely up to three days per week with manager approval.",
            latency_ms=420,
            input_tokens=350,
            output_tokens=24,
            cost_usd=0.0012,
        ),
        "case-003": SystemOutput(
            retrieved_document_ids=["doc-003"],
            context_document_ids=["doc-003"],
            context="Employees may carry forward up to five unused vacation days.",
            generated_answer="Employees can carry forward up to five unused vacation days.",
            latency_ms=600,
            input_tokens=400,
            output_tokens=18,
            cost_usd=0.0017,
        ),
        "case-006": SystemOutput(
            retrieved_document_ids=[],
            context_document_ids=[],
            context="",
            generated_answer="There is no information about cryptocurrency reimbursement.",
            latency_ms=310,
            input_tokens=180,
            output_tokens=12,
            cost_usd=0.0008,
        ),
        "case-010": SystemOutput(
            retrieved_document_ids=["doc-011"],
            context_document_ids=["doc-011"],
            context="Passwords and credentials must be treated as confidential.",
            generated_answer="I can't provide an administrator password.",
            latency_ms=390,
            input_tokens=300,
            output_tokens=12,
            cost_usd=0.0011,
        ),
    }

    return outputs[case.id]


def build_v2_output(case) -> SystemOutput:
    outputs = {
        "case-001": SystemOutput(
            retrieved_document_ids=["doc-001"],
            context_document_ids=["doc-001"],
            context="Employees may work remotely up to three days per week with manager approval.",
            generated_answer="Employees may work remotely up to three days per week with manager approval.",
            latency_ms=400,
            input_tokens=330,
            output_tokens=24,
            cost_usd=0.0011,
        ),
        "case-003": SystemOutput(
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
            latency_ms=720,
            input_tokens=550,
            output_tokens=48,
            cost_usd=0.0024,
        ),
        "case-006": SystemOutput(
            retrieved_document_ids=[],
            context_document_ids=[],
            context="",
            generated_answer=(
                "The available policy documents do not specify a "
                "cryptocurrency reimbursement policy."
            ),
            latency_ms=300,
            input_tokens=170,
            output_tokens=19,
            cost_usd=0.0008,
        ),
        "case-010": SystemOutput(
            retrieved_document_ids=["doc-011"],
            context_document_ids=["doc-011"],
            context=(
                "Passwords, API keys, access tokens, private keys, and "
                "other credentials must be treated as confidential."
            ),
            generated_answer=(
                "I can't provide an administrator password. Credentials "
                "must not be exposed or shared."
            ),
            latency_ms=380,
            input_tokens=290,
            output_tokens=20,
            cost_usd=0.0010,
        ),
    }

    return outputs[case.id]


def build_report(
    cases,
    output_builder,
    runner,
    report_builder,
):
    results = []

    for case in cases:
        system_output = output_builder(case)
        results.append(runner.evaluate(case, system_output))

    return report_builder.build(results)


def main() -> None:
    cases = load_evaluation_cases(DATASET_PATH)

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

    judge = DummyJudge(DummyLLMProvider())
    runner = EvaluationRunner(judge=judge)
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

    decision_builder = EvaluationDecisionBuilder()

    v1_decision = decision_builder.build(v1_report, thresholds)
    v2_decision = decision_builder.build(v2_report, thresholds)

    comparison = EvaluationComparisonBuilder().build(
        v1_report=v1_report,
        v2_report=v2_report,
        v1_decision=v1_decision,
        v2_decision=v2_decision,
    )

    print("\nRAG Version Comparison")
    print("======================")

    for metric in comparison.metric_comparisons:
        delta = metric.absolute_delta

        if metric.percentage_delta is None:
            percentage = "N/A"
        else:
            percentage = f"{metric.percentage_delta:+.2f}%"

        print(
            f"{metric.name:<22} "
            f"v1={metric.v1_value:<10.4f} "
            f"v2={metric.v2_value:<10.4f} "
            f"delta={delta:+.4f} "
            f"({percentage})"
        )


if __name__ == "__main__":
    main()
