def context_recall(
    relevant_context_ids: list[str],
    context_ids: list[str],
) -> float:
    if not relevant_context_ids or not context_ids:
        return 0.0

    relevant = set(relevant_context_ids)

    context = set(context_ids)

    return len(relevant & context) / len(relevant)


def context_precision(
    relevant_context_ids: list[str],
    context_ids: list[str],
) -> float:
    if not relevant_context_ids or not context_ids:
        return 0.0

    relevant = set(relevant_context_ids)

    context = set(context_ids)

    return len(relevant & context) / len(context)
