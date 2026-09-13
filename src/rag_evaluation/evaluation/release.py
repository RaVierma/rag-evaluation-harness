from enum import Enum

from rag_evaluation.models import (
    EvaluationDecision,
    RegressionDecision,
)


class ReleaseStatus(str, Enum):
    RELEASE = "release"
    REVIEW = "review"
    REJECT = "reject"


def determine_release_status(
    v2_decision: EvaluationDecision,
    regression_decision: RegressionDecision,
) -> ReleaseStatus:
    if v2_decision.is_passed and regression_decision.is_passed:
        return ReleaseStatus.RELEASE

    return ReleaseStatus.REVIEW
