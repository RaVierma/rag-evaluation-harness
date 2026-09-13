import pytest

from rag_evaluation.evaluation.runner import EvaluationRunner
from rag_evaluation.judges.llm import LLMJudge
from rag_evaluation.models import (
    EvaluationCase,
    EvaluationCaseCategory,
    GenerationEvaluation,
    RAGPerformance,
    SystemOutput,
)
from rag_evaluation.providers.generation import DummyLLMProvider

provider = DummyLLMProvider()
judge = LLMJudge(provider)
eval_runner = EvaluationRunner(judge)


def test_evaluation_runner():
    eval_case = EvaluationCase(
        id="case-0001",
        question="What is Rag?",
        expected_answer="Rag is a retrieval augmented generation model.",
        relevant_document_ids=["AA", "BB", "CC"],
        category=EvaluationCaseCategory.SIMPLE,
        relevance_judgments=[],
    )

    system_output = SystemOutput(
        retrieved_document_ids=["AA", "XX", "CC", "YY"],
        context_document_ids=["AA", "CC"],
        context="Rag is a retrieval augmented generation",
        generated_answer="Rag standard for retreival augmented gneration.",
    )

    rag_performance = RAGPerformance(
        total_latency_ms=100.0,
        retrieval_latency_ms=20.0,
        context_latency_ms=5.0,
        llm_latency_ms=75.0,
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.001,
    )

    result = eval_runner.evaluate(eval_case, system_output, rag_performance)

    assert isinstance(result.generation, GenerationEvaluation)

    assert result.case_id == eval_case.id
    assert result.system_output == system_output

    assert result.retrieval_recall == pytest.approx(2 / 3)
    assert result.retrieval_precision == pytest.approx(2 / 4)
    assert result.retrieval_rr == 1.0
    assert result.context_recall == pytest.approx(2 / 3)
    assert result.context_precision == pytest.approx(2 / 2)
