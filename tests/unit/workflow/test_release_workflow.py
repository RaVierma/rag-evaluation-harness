from rag_evaluation.evaluation.release import ReleaseStatus, determine_release_status
from rag_evaluation.models import (
    EvaluationDecision,
    RegressionDecision,
)


def test_release_when_quality_and_regression_gates_pass() -> None:
    quality_decision = EvaluationDecision(
        is_passed=True,
        passed=[],
        failures=[],
    )

    regression_decision = RegressionDecision(
        is_passed=True,
        passed=[],
        failures=[],
    )

    status = determine_release_status(
        v2_decision=quality_decision,
        regression_decision=regression_decision,
    )

    assert status == ReleaseStatus.RELEASE


def test_review_when_quality_gate_fails() -> None:
    quality_decision = EvaluationDecision(
        is_passed=False,
        passed=[],
        failures=[],
    )

    regression_decision = RegressionDecision(
        is_passed=True,
        passed=[],
        failures=[],
    )

    status = determine_release_status(
        v2_decision=quality_decision,
        regression_decision=regression_decision,
    )

    assert status == ReleaseStatus.REVIEW


def test_review_when_regression_gate_fails() -> None:
    quality_decision = EvaluationDecision(
        is_passed=True,
        passed=[],
        failures=[],
    )

    regression_decision = RegressionDecision(
        is_passed=False,
        passed=[],
        failures=[],
    )

    status = determine_release_status(
        v2_decision=quality_decision,
        regression_decision=regression_decision,
    )

    assert status == ReleaseStatus.REVIEW


def test_review_when_both_gates_fail() -> None:
    quality_decision = EvaluationDecision(
        is_passed=False,
        passed=[],
        failures=[],
    )

    regression_decision = RegressionDecision(
        is_passed=False,
        passed=[],
        failures=[],
    )

    status = determine_release_status(
        v2_decision=quality_decision,
        regression_decision=regression_decision,
    )

    assert status == ReleaseStatus.REVIEW
