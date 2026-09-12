from rag_evaluation.models.reports import RAGDiagnosis


def diagnose(
    retrieval_recall: float,
    groundedness: float,
    correctness: float,
    relevance: float,
) -> RAGDiagnosis:
    generation_good = groundedness == 1.0 and correctness == 1.0 and relevance == 1.0

    retrieval_good = retrieval_recall == 1.0

    if retrieval_good and generation_good:
        return RAGDiagnosis.PASS

    if retrieval_good and not generation_good:
        return RAGDiagnosis.GENERATION_JUDGE_UNCERTAIN

    if not retrieval_good and generation_good:
        return RAGDiagnosis.LUCKY

    return RAGDiagnosis.RETRIEVAL_FAILURE
