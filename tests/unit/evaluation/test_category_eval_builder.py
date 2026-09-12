import random

import pytest

from rag_evaluation.evaluation.category_report import CategoryEvaluationBuilder
from rag_evaluation.evaluation.runner import EvaluationRunner
from rag_evaluation.judges.llm import LLMJudge
from rag_evaluation.models import EvaluationCase
from rag_evaluation.models.cases import EvaluationCaseCategory
from rag_evaluation.models.outputs import SystemOutput
from rag_evaluation.providers.generation.dummy import DummyLLMProvider


@pytest.fixture
def eval_case_and_results():
    cases = []
    eval_results = []
    provider = DummyLLMProvider()
    judge = LLMJudge(provider)
    eval_runner = EvaluationRunner(judge)

    for i in range(1, 21):
        eval_case = EvaluationCase(
            id=f"case-000{i}",
            question="What is Rag?",
            expected_answer="Rag is a retrieval augmented generation model.",
            relevant_document_ids=["A", "B", "C"],
            category=random.choice(
                [
                    EvaluationCaseCategory.SIMPLE,
                    EvaluationCaseCategory.DIFFICULT,
                    EvaluationCaseCategory.MULTI_DOCUMENT,
                ]
            ),
        )

        cases.append(eval_case)

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

    return {"cases": cases, "eval_results": eval_results}


def test_category_evaluation_builder_for_success(eval_case_and_results):
    cases = eval_case_and_results["cases"]
    eval_results = eval_case_and_results["eval_results"]

    report_builder = CategoryEvaluationBuilder()

    build_result = report_builder.build(cases, eval_results)

    assert build_result.total_cases == len(cases)


def test_category_evaluation_builder_for_empty_cases_and_eval_result():

    with pytest.raises(ValueError):
        report_builder = CategoryEvaluationBuilder()

        report_builder.build([], [])
