from enum import Enum

from pydantic import BaseModel


class EvaluationThresholds(BaseModel):
    min_retrieval_recall: float
    min_retrieval_precision: float
    min_retrieval_mrr: float
    min_context_precision: float
    min_groundedness: float
    min_correctness: float
    max_p95_latency_ms: float
    max_avg_cost_usd: float


class MetricDirection(str, Enum):
    MIN = "MIN"
    MAX = "MAX"


class Metric(BaseModel):
    name: str
    actual_score: float
    required_score: float
    direction: MetricDirection


class EvaluationDecision(BaseModel):
    is_passed: bool
    passed: list[Metric]
    failures: list[Metric]


class RegressionThresholds(BaseModel):
    min_retrieval_recall_drop: float
    min_retrieval_precision_drop: float

    min_context_recall_drop: float
    min_context_precision_drop: float

    min_groundedness_drop: float
    min_correctness_drop: float
    min_relevance_drop: float

    max_p95_latency_ms_up: float
    max_avg_cost_usd_up: float


class RegressionMetric(BaseModel):
    name: str
    v1_value: float
    v2_value: float
    actual_delta: float
    threshold_delta: float
    direction: MetricDirection


class RegressionDecision(BaseModel):
    is_passed: bool
    passed: list[RegressionMetric]
    failures: list[RegressionMetric]
