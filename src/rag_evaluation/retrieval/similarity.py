import math


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
