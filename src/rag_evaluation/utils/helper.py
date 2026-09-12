import math
import statistics


def load_eval_prompt(
    version: str,
    question: str,
    context: str,
    expected_answer: str,
    generated_answer: str,
) -> str:
    with open(f"src/rag_evaluation/prompts/evaluation/{version}.txt", "r+") as f:
        return (
            f.read()
            .replace("{##question##}", question)
            .replace("{##context##}", context)
            .replace("{##expected_answer##}", expected_answer)
            .replace("{##generated_answer##}", generated_answer)
        )


def load_generation_prompt(version: str, question: str, context: str) -> str:
    with open(f"src/rag_evaluation/prompts/generation/{version}.txt", "r+") as f:
        return (
            f.read()
            .replace("{##question##}", question)
            .replace("{##context##}", context)
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


def cosine_similarity(
    a: tuple[float, ...],
    b: tuple[float, ...],
) -> float:
    if not a or not b:
        raise ValueError("a and b must not be empty")

    if len(a) != len(b):
        raise ValueError("a and b mismatch")

    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("vector must have not all zero value.")

    return dot_product / (norm_a * norm_b)
