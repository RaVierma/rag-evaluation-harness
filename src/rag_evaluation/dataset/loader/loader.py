import json
from pathlib import Path

from rag_evaluation.models import EvaluationCase


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases: list[EvaluationCase] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                cases.append(EvaluationCase.model_validate(data))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid evaluation case at line {line_number}"
                ) from exc

    return cases
