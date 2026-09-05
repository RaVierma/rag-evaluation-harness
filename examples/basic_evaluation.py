from pathlib import Path

from rag_evaluation.dataset import load_evaluation_cases
from rag_evaluation.evaluation.report import EvaluationReportBuilder
from rag_evaluation.evaluation.runner import EvaluationRunner
from rag_evaluation.judges.dummy import DummyJudge
from rag_evaluation.models import SystemOutput
from rag_evaluation.providers.dummy import DummyLLMProvider

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "examples" / "sample_cases.jsonl"


def build_system_output(case) -> SystemOutput:
    outputs = {
        "case-001": SystemOutput(
            retrieved_document_ids=["doc-001"],
            context_document_ids=["doc-001"],
            context=(
                "Employees may work remotely up to three days per week "
                "with manager approval."
            ),
            generated_answer=(
                "Employees may work remotely up to three days per week "
                "with manager approval."
            ),
            latency_ms=420.0,
            input_tokens=350,
            output_tokens=24,
            cost_usd=0.0012,
        ),
        "case-003": SystemOutput(
            retrieved_document_ids=["doc-003", "doc-004"],
            context_document_ids=["doc-003", "doc-004"],
            context=(
                "Employees may carry forward up to five unused vacation "
                "days. Unused days above the five-day limit normally expire "
                "unless an approved exception applies."
            ),
            generated_answer=(
                "Unused vacation days above the five-day carryover limit "
                "normally expire at year-end. Exceptions may be granted "
                "when business circumstances prevented the employee from "
                "taking approved leave."
            ),
            latency_ms=680.0,
            input_tokens=520,
            output_tokens=48,
            cost_usd=0.0021,
        ),
        "case-006": SystemOutput(
            retrieved_document_ids=[],
            context_document_ids=[],
            context="",
            generated_answer=(
                "The available policy documents do not specify a "
                "cryptocurrency reimbursement policy."
            ),
            latency_ms=310.0,
            input_tokens=180,
            output_tokens=19,
            cost_usd=0.0008,
        ),
        "case-010": SystemOutput(
            retrieved_document_ids=["doc-011"],
            context_document_ids=["doc-011"],
            context=(
                "Passwords, API keys, access tokens, private keys, and "
                "other credentials must be treated as confidential. "
                "Administrator passwords must not be exposed or shared."
            ),
            generated_answer=(
                "I can't provide an administrator password. Credentials "
                "must not be exposed or shared through unauthorized "
                "channels."
            ),
            latency_ms=390.0,
            input_tokens=300,
            output_tokens=27,
            cost_usd=0.0011,
        ),
    }

    try:
        return outputs[case.id]
    except KeyError as exc:
        raise ValueError(f"No system output defined for {case.id}") from exc


def main() -> None:
    cases = load_evaluation_cases(DATASET_PATH)

    provider = DummyLLMProvider()
    judge = DummyJudge(provider)
    runner = EvaluationRunner(judge=judge)

    results = []

    for case in cases:
        system_output = build_system_output(case)
        result = runner.evaluate(case, system_output)
        results.append(result)

    report = EvaluationReportBuilder().build(results)

    print("\nRAG Evaluation Report")
    print("====================")
    print(f"Total cases:         {report.total_cases}")
    print(f"Retrieval Recall:    {report.retrieval_recall:.2f}")
    print(f"Retrieval Precision: {report.retrieval_precision:.2f}")
    print(f"Retrieval MRR:       {report.retrieval_mrr:.2f}")
    print(f"Context Recall:      {report.context_recall:.2f}")
    print(f"Context Precision:   {report.context_precision:.2f}")
    print(f"Groundedness:        {report.groundedness:.2f}")
    print(f"Correctness:         {report.correctness:.2f}")
    print(f"Relevance:           {report.relevance:.2f}")
    print(f"Avg Latency:         {report.avg_latency_ms:.2f} ms")
    print(f"P95 Latency:         {report.p95_latency_ms:.2f} ms")
    print(f"Avg Input Tokens:    {report.avg_input_tokens:.2f}")
    print(f"Avg Output Tokens:   {report.avg_output_tokens:.2f}")
    print(f"Avg Cost:            ${report.avg_cost_usd:.4f}")


if __name__ == "__main__":
    main()
