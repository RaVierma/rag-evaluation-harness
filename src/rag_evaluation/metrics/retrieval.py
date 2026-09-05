def recall_at_k(
    relevant_document_ids: list[str],
    retrieved_document_ids: list[str],
    k: int,
) -> float:
    if not relevant_document_ids or not retrieved_document_ids:
        return 0.0

    relevant = set(relevant_document_ids)
    retrieved = retrieved_document_ids[:k]

    relevant_retrieved = sum(doc_id in relevant for doc_id in retrieved)

    return relevant_retrieved / len(relevant)


def precision_at_k(
    relevant_document_ids: list[str],
    retrieved_document_ids: list[str],
    k: int,
) -> float:
    if not relevant_document_ids or not retrieved_document_ids or k <= 0:
        return 0.0

    relevant = set(relevant_document_ids)
    retrieved = retrieved_document_ids[:k]

    relevant_retrieved = sum(doc_id in relevant for doc_id in retrieved)

    return relevant_retrieved / len(retrieved)


def reciprocal_rank(
    relevant_document_ids: list[str],
    retrieved_document_ids: list[str],
) -> float:
    if not relevant_document_ids or not retrieved_document_ids:
        return 0.0

    relevant = set(relevant_document_ids)

    for rank, doc_id in enumerate(retrieved_document_ids, 1):
        if doc_id in relevant:
            return 1 / rank

    return 0.0


def mean_reciprocal_rank(
    reciprocal_ranks: list[float],
) -> float:
    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
