from typing import Literal

from pydantic import BaseModel

from common.types import NonEmptyString

Score = Literal[0.0, 0.5, 1.0]


class Evidence(BaseModel):
    claim: NonEmptyString
    support: NonEmptyString


class MetricEvaluation(BaseModel):
    score: Score  # type: ignore
    reason: NonEmptyString
    evidence: list[Evidence]


class GenerationEvaluation(BaseModel):
    groundedness: MetricEvaluation
    correctness: MetricEvaluation
    relevance: MetricEvaluation
