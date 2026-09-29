import json

import pytest

from evaluation.models.artifacts import EvaluationArtifact
from evaluation.models.reports import EvaluationReport


def build_report() -> EvaluationReport:
    return EvaluationReport(
        total_cases=22,
        retrieval_recall=0.96,
        retrieval_precision=0.91,
        retrieval_mrr=0.94,
        context_recall=0.93,
        context_precision=0.89,
        groundedness=0.94,
        correctness=0.92,
        relevance=0.95,
        avg_latency_ms=2100.0,
        p95_latency_ms=2800.0,
        avg_input_tokens=3200.0,
        avg_output_tokens=450.0,
        avg_cost_usd=0.025,
        case_results=[],
    )


def test_from_report_maps_metadata() -> None:
    report = build_report()

    artifact = EvaluationArtifact.from_report(
        version="v2",
        dataset="rag-eval-v1",
        report=report,
    )

    assert artifact.version == "v2"
    assert artifact.dataset == "rag-eval-v1"
    assert artifact.total_cases == 22


def test_from_report_maps_all_aggregate_metrics() -> None:
    report = build_report()

    artifact = EvaluationArtifact.from_report(
        version="v2",
        dataset="rag-eval-v1",
        report=report,
    )

    assert artifact.metrics.retrieval_recall == 0.96
    assert artifact.metrics.retrieval_precision == 0.91
    assert artifact.metrics.retrieval_mrr == 0.94

    assert artifact.metrics.context_recall == 0.93
    assert artifact.metrics.context_precision == 0.89

    assert artifact.metrics.groundedness == 0.94
    assert artifact.metrics.correctness == 0.92
    assert artifact.metrics.relevance == 0.95

    assert artifact.metrics.avg_latency_ms == 2100.0
    assert artifact.metrics.p95_latency_ms == 2800.0
    assert artifact.metrics.avg_input_tokens == 3200.0
    assert artifact.metrics.avg_output_tokens == 450.0
    assert artifact.metrics.avg_cost_usd == 0.025


def test_case_results_are_not_in_artifact() -> None:
    report = build_report()

    artifact = EvaluationArtifact.from_report(
        version="v2",
        dataset="rag-eval-v1",
        report=report,
    )

    assert "case_results" not in artifact.model_fields_set


def test_artifact_serializes_to_json() -> None:
    report = build_report()

    artifact = EvaluationArtifact.from_report(
        version="v2",
        dataset="rag-eval-v1",
        report=report,
    )

    payload = artifact.model_dump_json()

    data = json.loads(payload)

    assert data["version"] == "v2"
    assert data["dataset"] == "rag-eval-v1"
    assert data["total_cases"] == 22

    assert data["metrics"]["correctness"] == 0.92
    assert data["metrics"]["groundedness"] == 0.94
    assert data["metrics"]["p95_latency_ms"] == 2800.0


def test_artifact_does_not_serialize_case_results() -> None:
    report = build_report()

    artifact = EvaluationArtifact.from_report(
        version="v2",
        dataset="rag-eval-v1",
        report=report,
    )

    data = json.loads(artifact.model_dump_json())

    assert "case_results" not in data


def test_from_report_preserves_zero_metrics() -> None:
    report = EvaluationReport(
        total_cases=1,
        retrieval_recall=0.0,
        retrieval_precision=0.0,
        retrieval_mrr=0.0,
        context_recall=0.0,
        context_precision=0.0,
        groundedness=0.0,
        correctness=0.0,
        relevance=0.0,
        avg_latency_ms=0.0,
        p95_latency_ms=0.0,
        avg_input_tokens=0.0,
        avg_output_tokens=0.0,
        avg_cost_usd=0.0,
        case_results=[],
    )

    artifact = EvaluationArtifact.from_report(
        version="v1",
        dataset="empty-quality-dataset",
        report=report,
    )

    assert artifact.metrics.correctness == 0.0
    assert artifact.metrics.groundedness == 0.0
    assert artifact.metrics.p95_latency_ms == 0.0
    assert artifact.metrics.avg_cost_usd == 0.0
