from evaluation.diagnosis import RAGDiagnosis, diagnose


def test_diagnose_pass():
    result = diagnose(
        retrieval_recall=1.0,
        groundedness=1.0,
        correctness=1.0,
        relevance=1.0,
    )

    assert result == RAGDiagnosis.PASS


def test_diagnose_generation_judge_uncertain():
    result = diagnose(
        retrieval_recall=1.0,
        groundedness=0.5,
        correctness=1.0,
        relevance=1.0,
    )

    assert result == RAGDiagnosis.GENERATION_JUDGE_UNCERTAIN


def test_diagnose_lucky():
    result = diagnose(
        retrieval_recall=0.8,
        groundedness=1.0,
        correctness=1.0,
        relevance=1.0,
    )

    assert result == RAGDiagnosis.LUCKY


def test_diagnose_retrieval_failure():
    result = diagnose(
        retrieval_recall=0.8,
        groundedness=0.5,
        correctness=1.0,
        relevance=1.0,
    )

    assert result == RAGDiagnosis.RETRIEVAL_FAILURE
