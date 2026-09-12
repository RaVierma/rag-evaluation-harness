import pytest

from rag_evaluation.evaluation.runner import EvaluationRunner
from rag_evaluation.judges.llm import LLMJudge
from rag_evaluation.models import (
    EvaluationCase,
    EvaluationCaseCategory,
    GenerationEvaluation,
    SystemOutput,
)
from rag_evaluation.providers.generation.dummy import DummyLLMProvider

provider = DummyLLMProvider()
judge = LLMJudge(provider)
eval_runner = EvaluationRunner(judge)


def test_evaluation_runner():
    eval_case = EvaluationCase(
        id="case-0001",
        question="What is Rag?",
        expected_answer="Rag is a retrieval augmented generation model.",
        relevant_document_ids=["A", "B", "C"],
        category=EvaluationCaseCategory.SIMPLE,
    )

    system_output = SystemOutput(
        retrieved_document_ids=["A", "X", "C", "Y"],
        context_document_ids=["A", "C"],
        context="Rag is a retrieval augmented generation",
        generated_answer="Rag standard for retreival augmented gneration.",
        latency_ms=0.0,  # dummy
        input_tokens=15,  # dummy
        output_tokens=6,  # dummy
        cost_usd=0.013,  # dummy
    )

    result = eval_runner.evaluate(eval_case, system_output)

    assert isinstance(result.generation, GenerationEvaluation)

    assert result.case_id == eval_case.id
    assert result.system_output == system_output

    assert result.retrieval_recall == pytest.approx(2 / 3)
    assert result.retrieval_precision == pytest.approx(2 / 4)
    assert result.retrieval_rr == 1.0
    assert result.context_recall == pytest.approx(2 / 3)
    assert result.context_precision == pytest.approx(2 / 2)
