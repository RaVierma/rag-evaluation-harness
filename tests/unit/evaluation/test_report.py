import random

import pytest

from rag_evaluation.evaluation.report import EvaluationReportBuilder
from rag_evaluation.evaluation.runner import EvaluationRunner
from rag_evaluation.judges.dummy import DummyJudge
from rag_evaluation.models import (
    EvaluationCase,
    EvaluationCaseCategory,
    SystemOutput,
)
from rag_evaluation.providers.dummy import DummyLLMProvider


@pytest.fixture
def eval_case_results():
    eval_results = []
    provider = DummyLLMProvider()
    judge = DummyJudge(provider)
    eval_runner = EvaluationRunner(judge)

    for i in range(1, 6):
        eval_case = EvaluationCase(
            id=f"case-000{i}",
            question="What is Rag?",
            expected_answer="Rag is a retrieval augmented generation model.",
            relevant_document_ids=["A", "B", "C"],
            category=random.choice(
                [EvaluationCaseCategory.SIMPLE, EvaluationCaseCategory.DIFFICULT]
            ),
        )

        retrieved_document_ids = ["A", "X", "C", "Y"]
        random.shuffle(retrieved_document_ids)

        system_output = SystemOutput(
            retrieved_document_ids=retrieved_document_ids,
            context_document_ids=["A", "C"],
            context="Rag is a retrieval augmented generation",
            generated_answer="Rag standard for retreival augmented gneration.",
            latency_ms=random.choice([100, 200, 500, 250]),  # dummy
            input_tokens=random.choice([50, 100, 60, 70]),  # dummy
            output_tokens=random.choice([100, 150, 200, 130]),  # dummy
            cost_usd=random.choice([0.01, 0.02, 0.015]),  # dummy
        )
        result = eval_runner.evaluate(eval_case, system_output)

        eval_results.append(result)

    return eval_results


def test_evaluation_report(eval_case_results):

    report_builder = EvaluationReportBuilder()

    build_result = report_builder.build(eval_case_results)

    assert build_result.total_cases == len(eval_case_results)


def test_evaluation_report_no_eval_case_result():

    with pytest.raises(ValueError):
        report_builder = EvaluationReportBuilder()

        report_builder.build([])
