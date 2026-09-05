import pytest

from rag_evaluation.metrics.context import context_precision, context_recall


# Context Recall Test
def test_context_recall_for_empty_relevant_context():
    relevant_context_ids = []
    context_ids = ["A", "B"]

    assert context_recall(relevant_context_ids, context_ids) == 0.0


def test_context_recall_for_empty_supplied_context():
    relevant_context_ids = ["A", "B"]
    context_ids = []

    assert context_recall(relevant_context_ids, context_ids) == 0.0


def test_context_recall_for_no_relevant_context_found():
    relevant_context_ids = ["A", "B"]
    context_ids = ["X", "Y"]

    assert context_recall(relevant_context_ids, context_ids) == 0.0


def test_context_recall_for_all_relevant_context_found():
    relevant_context_ids = ["A", "B"]
    context_ids = ["A", "B"]

    assert context_recall(relevant_context_ids, context_ids) == 1.0


def test_context_recall_for_partial_context_found():
    relevant_context_ids = ["A", "B"]
    context_ids = ["A", "C"]

    assert context_recall(relevant_context_ids, context_ids) == 0.5


def test_context_recall_for_fractional_result():
    relevant_context_ids = ["A", "B", "C"]
    context_ids = ["A", "X", "C", "Y"]

    assert context_recall(
        relevant_context_ids,
        context_ids,
    ) == pytest.approx(2 / 3)


# Context Precision Test
def test_context_precision_for_empty_relevant_context():
    relevant_context_ids = []
    context_ids = ["A", "B"]

    assert context_precision(relevant_context_ids, context_ids) == 0.0


def test_context_precision_for_empty_supplied_context():
    relevant_context_ids = ["A", "B"]
    context_ids = []

    assert context_precision(relevant_context_ids, context_ids) == 0.0


def test_context_precision_for_no_relevant_context_found():
    relevant_context_ids = ["A", "B"]
    context_ids = ["X", "Y"]

    assert context_precision(relevant_context_ids, context_ids) == 0.0


def test_context_precision_for_all_relevant_context_found():
    relevant_context_ids = ["A", "B"]
    context_ids = ["A", "B"]

    assert context_precision(relevant_context_ids, context_ids) == 1.0


def test_context_precision_for_partial_context_found():
    relevant_context_ids = ["A", "B"]
    context_ids = ["A", "C"]

    assert context_precision(relevant_context_ids, context_ids) == 0.5


def test_context_precision_for_fractional_result():
    relevant_context_ids = ["A", "B", "C"]
    context_ids = ["A", "X", "C", "Y"]

    assert context_precision(
        relevant_context_ids,
        context_ids,
    ) == pytest.approx(2 / 4)
