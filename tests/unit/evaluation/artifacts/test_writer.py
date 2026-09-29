import json

from evaluation.artifacts.writer import write_evaluation_artifact
from evaluation.models.artifacts import EvaluationArtifact, EvaluationMetrics


def build_artifact() -> EvaluationArtifact:
    return EvaluationArtifact(
        version="v2",
        dataset="rag-eval-v1",
        total_cases=22,
        metrics=EvaluationMetrics(
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
        ),
    )


def test_write_evaluation_artifact_creates_file(tmp_path) -> None:
    artifact = build_artifact()
    output_path = tmp_path / "evaluation-result.json"

    write_evaluation_artifact(
        artifact=artifact,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.is_file()


def test_write_evaluation_artifact_writes_valid_json(tmp_path) -> None:
    artifact = build_artifact()
    output_path = tmp_path / "evaluation-result.json"

    write_evaluation_artifact(
        artifact=artifact,
        output_path=output_path,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["version"] == "v2"
    assert data["dataset"] == "rag-eval-v1"
    assert data["total_cases"] == 22
    assert data["metrics"]["correctness"] == 0.92
    assert data["metrics"]["groundedness"] == 0.94
    assert data["metrics"]["p95_latency_ms"] == 2800.0


def test_write_evaluation_artifact_creates_parent_directories(
    tmp_path,
) -> None:
    artifact = build_artifact()

    output_path = (
        tmp_path / "artifacts" / "evaluation" / "run-1" / "evaluation-result.json"
    )

    write_evaluation_artifact(
        artifact=artifact,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.is_file()


def test_write_evaluation_artifact_does_not_include_case_results(
    tmp_path,
) -> None:
    artifact = build_artifact()
    output_path = tmp_path / "evaluation-result.json"

    write_evaluation_artifact(
        artifact=artifact,
        output_path=output_path,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert "case_results" not in data
