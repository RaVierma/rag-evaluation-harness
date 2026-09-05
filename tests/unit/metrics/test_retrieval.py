import pytest

from rag_evaluation.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# Recall Test
def test_recall_at_k_for_all_relevant():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "B", "C"]

    assert recall_at_k(relevant, retrieved, 3) == 1.0


def test_recall_at_k_for_some_relevant():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "X", "C"]

    assert recall_at_k(relevant, retrieved, 3) == 0.6666666666666666


def test_recall_at_k_for_no_doc_retrieved():
    relevant = ["A", "B", "C"]
    retrieved = []

    assert recall_at_k(relevant, retrieved, 3) == 0.0


def test_recall_at_k_for_k_less_than_len_retrieved():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "B", "C", "D"]

    assert recall_at_k(relevant, retrieved, 3) == 1.0


def test_recall_at_k_for_k_greater_than_len_retrieved():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "B", "D"]

    assert recall_at_k(relevant, retrieved, 5) == 0.6666666666666666


def test_recall_at_k_for_empty_inputs():
    relevant = []
    retrieved = []

    assert recall_at_k(relevant, retrieved, 5) == 0.0


# Precision Tests
def test_precision_at_k_for_all_relevant():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "B", "C"]

    assert precision_at_k(relevant, retrieved, 3) == 1.0


def test_precision_at_k_for_some_relevant():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "X", "C"]

    assert precision_at_k(relevant, retrieved, 3) == 0.6666666666666666


def test_precision_at_k_for_no_doc_retrieved():
    relevant = ["A", "B", "C"]
    retrieved = []

    assert precision_at_k(relevant, retrieved, 3) == 0.0


def test_precision_at_k_for_k_greater_than_len_retrieved():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "B", "D"]

    assert precision_at_k(relevant, retrieved, 5) == 0.6666666666666666


def test_precision_at_k_for_k_less_than_equal_zero():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "B", "C"]

    assert precision_at_k(relevant, retrieved, -1) == 0.0


def test_precision_at_k_when_k_less_than_retrieved():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "X", "B", "C"]

    assert precision_at_k(relevant, retrieved, 2) == pytest.approx(1 / 2)


def test_precision_at_k_for_empty_inputs():
    relevant = []
    retrieved = []

    assert precision_at_k(relevant, retrieved, 5) == 0.0


# Reciprocal Rank Test
def test_reciprocal_rank_for_relevant_doc_at_rank_1():
    relevant = ["A", "B", "C"]
    retrieved = ["A", "C", "X"]

    assert reciprocal_rank(relevant, retrieved) == 1.0


def test_reciprocal_rank_for_relevant_doc_at_rank_2():
    relevant = ["A", "B", "C"]
    retrieved = ["D", "A", "B"]

    assert reciprocal_rank(relevant, retrieved) == 0.5


def test_reciprocal_rank_for_k_greater_than_len_retrieved():
    relevant = ["A", "B", "C"]
    retrieved = ["D", "B", "A"]

    assert reciprocal_rank(relevant, retrieved) == 0.5


def test_reciprocal_rank_for_no_doc_retrieved():
    relevant = ["A", "B", "C"]
    retrieved = []

    assert reciprocal_rank(relevant, retrieved) == 0.0


def test_reciprocal_rank_uses_first_relevant_in_retrieved_order():
    relevant = ["A", "B"]
    retrieved = ["B", "X", "A"]

    assert reciprocal_rank(relevant, retrieved) == 1.0


def test_reciprocal_rank_for_empty_inputs():
    relevant = []
    retrieved = []

    assert reciprocal_rank(relevant, retrieved) == 0.0


# MRR Tests
def test_mean_reciprocal_rank_for_empty_inputs():
    reciprocal_ranks = []

    assert mean_reciprocal_rank(reciprocal_ranks) == 0.0


def test_mean_reciprocal_rank_for_only_1_rerank():
    reciprocal_ranks = [1.0]

    assert mean_reciprocal_rank(reciprocal_ranks) == 1.0


def test_mean_reciprocal_rank_for_only_2_input():
    reciprocal_ranks = [1.0, 0.5]

    assert mean_reciprocal_rank(reciprocal_ranks) == 0.75


def test_mean_reciprocal_rank_for_multiple_inputs():
    reciprocal_ranks = [1.0, 0.5, 0.25, 0]

    assert mean_reciprocal_rank(reciprocal_ranks) == 0.4375
