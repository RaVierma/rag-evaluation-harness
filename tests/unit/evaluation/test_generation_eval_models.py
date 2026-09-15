import pytest
from pydantic import ValidationError

from evaluation.models.generation import (
    Evidence,
    GenerationEvaluation,
    MetricEvaluation,
)


def test_generation_eval_model_for_all_input_valid():

    groundedness = MetricEvaluation(
        score=1.0,
        reason="All factual claims in the generated answer are supported by the provided context.",
        evidence=[
            Evidence(
                claim="Employees can claim up to $500 per year.",
                support="Employees can claim up to $500 per year.",
            )
        ],
    )

    correctness = MetricEvaluation(
        score=1.0,
        reason="The generated answer matches the expected answer in meaning.",
        evidence=[
            Evidence(
                claim="The annual allowance is $500.",
                support="Employees can claim up to $500 per year.",
            )
        ],
    )

    relevance = MetricEvaluation(
        score=1.0,
        reason="The answer directly addresses the user's question about the allowance.",
        evidence=[
            Evidence(
                claim="The annual allowance is $500.",
                support="What is the company's work-from-home allowance?",
            )
        ],
    )

    GenerationEvaluation(
        groundedness=groundedness, correctness=correctness, relevance=relevance
    )


def test_generation_eval_model_for_invalid_score():

    with pytest.raises(
        expected_exception=(ValueError,), match=".*Input should be 0.0, 0.5 or 1.0"
    ):
        groundedness = MetricEvaluation(
            score=0.1,
            reason="All factual claims in the generated answer are supported by the provided context.",
            evidence=[
                Evidence(
                    claim="Employees can claim up to $500 per year.",
                    support="Employees can claim up to $500 per year.",
                )
            ],
        )

        correctness = MetricEvaluation(
            score=0.0,
            reason="The generated answer matches the expected answer in meaning.",
            evidence=[
                Evidence(
                    claim="The annual allowance is $500.",
                    support="Employees can claim up to $500 per year.",
                )
            ],
        )

        relevance = MetricEvaluation(
            score=0.7,
            reason="The answer directly addresses the user's question about the allowance.",
            evidence=[
                Evidence(
                    claim="The annual allowance is $500.",
                    support="What is the company's work-from-home allowance?",
                )
            ],
        )

        GenerationEvaluation(
            groundedness=groundedness, correctness=correctness, relevance=relevance
        )


def test_generation_eval_model_for_empty_reason():
    with pytest.raises(
        expected_exception=(ValueError),
        match=".*String should have at least 2 characters",
    ):
        groundedness = MetricEvaluation(
            score=1.0,
            reason="",
            evidence=[
                Evidence(
                    claim="Employees can claim up to $500 per year.",
                    support="Employees can claim up to $500 per year.",
                )
            ],
        )

        correctness = MetricEvaluation(
            score=0.0,
            reason="The generated answer matches the expected answer in meaning.",
            evidence=[
                Evidence(
                    claim="The annual allowance is $500.",
                    support="Employees can claim up to $500 per year.",
                )
            ],
        )

        relevance = MetricEvaluation(
            score=0.5,
            reason="The answer directly addresses the user's question about the allowance.",
            evidence=[
                Evidence(
                    claim="The annual allowance is $500.",
                    support="What is the company's work-from-home allowance?",
                )
            ],
        )

        GenerationEvaluation(
            groundedness=groundedness, correctness=correctness, relevance=relevance
        )


def test_generation_eval_model_for_no_evidences():
    groundedness = MetricEvaluation(
        score=1.0,
        reason="All factual claims in the generated answer are supported by the provided context.",
        evidence=[],
    )

    correctness = MetricEvaluation(
        score=0.0,
        reason="The generated answer matches the expected answer in meaning.",
        evidence=[],
    )

    relevance = MetricEvaluation(
        score=0.5,
        reason="The answer directly addresses the user's question about the allowance.",
        evidence=[],
    )

    GenerationEvaluation(
        groundedness=groundedness, correctness=correctness, relevance=relevance
    )


def test_generation_eval_model_for_invalid_nested_score():
    with pytest.raises(ValidationError):
        GenerationEvaluation(
            groundedness={
                "score": 0.7,
                "reason": "Invalid score",
                "evidence": [
                    {
                        "claim": "Some claim",
                        "support": "Some support",
                    }
                ],
            },
            correctness={
                "score": 1.0,
                "reason": "Correct",
                "evidence": [
                    {
                        "claim": "Some claim",
                        "support": "Some support",
                    }
                ],
            },
            relevance={
                "score": 1.0,
                "reason": "Relevant",
                "evidence": [
                    {
                        "claim": "Some claim",
                        "support": "Some support",
                    }
                ],
            },
        )
