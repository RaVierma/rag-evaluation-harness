import math


def dcg_at_k(
    relevance_scores: list[int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    return sum(
        (2**rel - 1) / math.log2(pos + 1)
        for pos, rel in enumerate(relevance_scores[:k], start=1)
    )


def ndcg_at_k(
    relevance_scores: list[int],
    k: int,
) -> float:
    if not relevance_scores:
        raise ValueError("relevance scores are required")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    actual_dcg = dcg_at_k(relevance_scores, k)

    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg
