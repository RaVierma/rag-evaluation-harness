import random

import pytest

from evaluation.reporting.category_report import CategoryEvaluationBuilder
from evaluation.judges import LLMJudge
from evaluation.models.cases import (
    EvaluationCase,
    EvaluationCaseCategory,
)
from evaluation.runner import EvaluationRunner
from rag.models.outputs import SystemOutput
from rag.models.performance import RAGPerformance
from rag.providers.generation import DummyLLMProvider


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
            relevant_document_ids=["AA", "BB", "CC"],
            relevance_judgments=[],
            category=random.choice(
                [
                    EvaluationCaseCategory.SIMPLE,
                    EvaluationCaseCategory.DIFFICULT,
                    EvaluationCaseCategory.MULTI_DOCUMENT,
                ]
            ),
        )

        cases.append(eval_case)

        retrieved_document_ids = ["AA", "XX", "CC", "YY"]
        random.shuffle(retrieved_document_ids)

        system_output = SystemOutput(
            retrieved_document_ids=retrieved_document_ids,
            context_document_ids=["AA", "CC"],
            context="Rag is a retrieval augmented generation",
            generated_answer="Rag standard for retreival augmented gneration.",
        )

        rag_performance = RAGPerformance(
            total_latency_ms=random.choice([100, 200, 500, 250]),
            retrieval_latency_ms=20.0,
            context_latency_ms=5.0,
            llm_latency_ms=75.0,
            input_tokens=random.choice([50, 100, 60, 70]),
            output_tokens=random.choice([100, 150, 200, 130]),
            cost_usd=random.choice([0.01, 0.02, 0.015]),
        )
        result = eval_runner.evaluate(eval_case, system_output, rag_performance)

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
