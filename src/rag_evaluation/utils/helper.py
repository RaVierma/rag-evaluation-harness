import statistics


def load_builder(
    version: str,
    question: str,
    context: str,
    expected_answer: str,
    generated_answer: str,
) -> str:
    with open(f"src/rag_evaluation/prompts/eval_prompt_{version}.txt", "r+") as f:
        return (
            f.read()
            .replace("{##question##}", question)
            .replace("{##context##}", context)
            .replace("{##expected_answer##}", expected_answer)
            .replace("{##generated_answer##}", generated_answer)
        )


def calculate_percentage_delta(v1, v2):
    if v1 == 0:
        return None
    else:
        return ((v2 - v1) / v1) * 100


def calculate_p95(values: list[float]) -> float:
    if not values:
        raise ValueError("values cannot be empty")

    if len(values) < 2:
        return values[0]

    return statistics.quantiles(values, n=100)[94]
