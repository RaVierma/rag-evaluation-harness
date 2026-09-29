from pathlib import Path

from evaluation.models.artifacts import EvaluationArtifact


def write_evaluation_artifact(
    artifact: EvaluationArtifact,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )
