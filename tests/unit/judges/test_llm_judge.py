import pytest

from rag_evaluation.judges import LLMJudge
from rag_evaluation.models import GenerationEvaluation
from rag_evaluation.providers.generation import DummyLLMProvider


def test_dummy_judge_for_success():
    provider = DummyLLMProvider()
    dummy_judge = LLMJudge(provider)
    result = dummy_judge.evaluate(
        question="What",
        context="this is context",
        expected_answer="great answer",
        generated_answer="gen answer",
    )

    assert isinstance(result, GenerationEvaluation)

    assert result.groundedness.score == 1.0
    assert result.correctness.score == 1.0
    assert result.relevance.score == 1.0


def test_dummy_judge_for_empty_question_expected_answer_generated_answer():
    with pytest.raises(ValueError):
        provider = DummyLLMProvider()
        dummy_judge = LLMJudge(provider)
        dummy_judge.evaluate(
            question="",
            context="this is context",
            expected_answer="",
            generated_answer="",
        )


def test_dummy_judge_for_provider_failure_invalid_json():
    with pytest.raises(ValueError, match="Judge raises controlled JSON error"):
        provider = DummyLLMProvider(response="{{}")
        dummy_judge = LLMJudge(provider)
        dummy_judge.evaluate(
            question="What",
            context="this is context",
            expected_answer="great answer",
            generated_answer="gen answer",
        )


def test_dummy_judge_for_provider_failure_due_invalid_eval():
    with pytest.raises(ValueError, match="Judge raises Pydantic ValidationError"):
        provider = DummyLLMProvider(
            response="""{
  "groundedness": {
    "score": 0.7,
    "reason": "..."
  }
}"""
        )
        dummy_judge = LLMJudge(provider)
        dummy_judge.evaluate(
            question="What",
            context="this is context",
            expected_answer="great answer",
            generated_answer="gen answer",
        )
