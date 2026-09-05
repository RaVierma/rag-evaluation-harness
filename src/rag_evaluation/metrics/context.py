def context_recall(
    relevant_context_ids: list[str],
    context_ids: list[str],
) -> float:
    if not relevant_context_ids or not context_ids:
        return 0.0

    relevant = set(relevant_context_ids)

    relevant_context_count = sum(context_id in relevant for context_id in context_ids)

    return relevant_context_count / len(relevant)


def context_precision(
    relevant_context_ids: list[str],
    context_ids: list[str],
) -> float:
    if not relevant_context_ids or not context_ids:
        return 0.0

    relevant = set(relevant_context_ids)

    relevant_context_count = sum(context_id in relevant for context_id in context_ids)

    return relevant_context_count / len(context_ids)
