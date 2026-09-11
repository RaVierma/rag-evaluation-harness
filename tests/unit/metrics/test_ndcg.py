import pytest

from rag_evaluation.metrics.ndcg import ndcg_at_k


def test_ndcg_at_k_for_pefect_ranking():
    relevance_scores = [3, 2, 1, 0]
    k = 4
    ndcg_score = ndcg_at_k(relevance_scores, k)

    assert ndcg_score == 1.0


def test_ndcg_at_k_for_impefect_ranking():
    relevance_scores = [3, 1, 2, 0]
    k = 4
    ndcg_score = ndcg_at_k(relevance_scores, k)

    assert ndcg_score < 1.0


def test_ndcg_at_k_for_top_k_one():
    relevance_scores = [3, 1, 2, 0]
    k = 1
    ndcg_score = ndcg_at_k(relevance_scores, k)

    assert ndcg_score == 1.0


def test_ndcg_at_k_for_all_zero():
    relevance_scores = [0, 0, 0]
    k = 1
    ndcg_score = ndcg_at_k(relevance_scores, k)

    assert ndcg_score == 0.0


def test_ndcg_at_k_for_empty_relevance_scores():
    with pytest.raises(ValueError):
        relevance_scores = []
        k = 1
        ndcg_at_k(relevance_scores, k)


def test_ndcg_at_k_for_empty_relevance_score():
    with pytest.raises(ValueError):
        relevance_scores = []
        k = 1
        ndcg_at_k(relevance_scores, k)


def test_ndcg_at_k_for_zero_top_k():
    with pytest.raises(ValueError):
        relevance_scores = [3, 2, 1, 0]
        k = 0
        ndcg_at_k(relevance_scores, k)


def test_ndcg_at_k_for_top_k_greater_than_list():
    relevance_scores = [3, 2, 1, 0]
    k = 10
    ndcg_at_k(relevance_scores, k)
