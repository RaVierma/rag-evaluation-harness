import pytest

from rag_evaluation.models import EmbeddedChunk, HybridRetrievedChunk
from rag_evaluation.reranking import DummyReranker


@pytest.fixture
def reranker() -> DummyReranker:
    return DummyReranker()


@pytest.fixture
def hybrid_candidates() -> list[HybridRetrievedChunk]:
    return [
        HybridRetrievedChunk(
            chunk=EmbeddedChunk(
                chunk_id="chunk-a",
                document_id="doc-a",
                content=(
                    "Managers make reasonable efforts "
                    "to accommodate annual leave requests."
                ),
                embedding=(0.1, 0.2),
            ),
            dense_score=0.80,
            bm25_score=5.0,
            dense_rank=1,
            bm25_rank=2,
            rrf_score=0.032,
        ),
        HybridRetrievedChunk(
            chunk=EmbeddedChunk(
                chunk_id="chunk-b",
                document_id="doc-b",
                content=(
                    "A manager may request an exception to the annual leave policy."
                ),
                embedding=(0.2, 0.3),
            ),
            dense_score=0.75,
            bm25_score=6.0,
            dense_rank=2,
            bm25_rank=1,
            rrf_score=0.031,
        ),
        HybridRetrievedChunk(
            chunk=EmbeddedChunk(
                chunk_id="chunk-c",
                document_id="doc-c",
                content=(
                    "Exceptions may be granted subject to manager and HR approval."
                ),
                embedding=(0.3, 0.4),
            ),
            dense_score=0.70,
            bm25_score=5.5,
            dense_rank=3,
            bm25_rank=3,
            rrf_score=0.030,
        ),
        HybridRetrievedChunk(
            chunk=EmbeddedChunk(
                chunk_id="chunk-d",
                document_id="doc-d",
                content=("Approved exceptions must be documented and submitted to HR."),
                embedding=(0.4, 0.5),
            ),
            dense_score=0.65,
            bm25_score=4.5,
            dense_rank=4,
            bm25_rank=4,
            rrf_score=0.029,
        ),
    ]


class TestTokenize:
    def test_tokenize_lowercases_text(self, reranker: DummyReranker):
        result = reranker._tokenize("Manager APPROVAL Exception")

        assert result == {"manager", "approval", "exception"}

    def test_tokenize_removes_punctuation(self, reranker: DummyReranker):
        result = reranker._tokenize("Who can approve an exception to the policy?")

        assert result == {"approve", "exception", "policy"}

    def test_tokenize_removes_stop_words(self, reranker: DummyReranker):
        result = reranker._tokenize("Who can approve an exception to the policy?")

        assert "who" not in result
        assert "can" not in result
        assert "an" not in result
        assert "to" not in result
        assert "the" not in result

    def test_tokenize_empty_text(self, reranker: DummyReranker):
        assert reranker._tokenize("") == set()

    def test_tokenize_whitespace(self, reranker: DummyReranker):
        assert reranker._tokenize("   ") == set()


class TestScore:
    def test_score_full_overlap(self, reranker: DummyReranker):
        score = reranker._score(
            "approve exception",
            "approve exception",
        )

        assert score == 1.0

    def test_score_partial_overlap(self, reranker: DummyReranker):
        score = reranker._score(
            "approve exception policy",
            "approve exception",
        )

        assert score == pytest.approx(2 / 3)

    def test_score_no_overlap(self, reranker: DummyReranker):
        score = reranker._score(
            "approve exception",
            "remote work policy",
        )

        assert score == 0.0

    def test_score_is_case_insensitive(self, reranker: DummyReranker):
        score = reranker._score(
            "Approve Exception",
            "approve exception",
        )

        assert score == 1.0

    def test_score_ignores_stop_words(self, reranker: DummyReranker):
        score = reranker._score(
            "Who can approve an exception?",
            "approve exception",
        )

        assert score == 1.0

    def test_score_empty_query_raises(self, reranker: DummyReranker):
        with pytest.raises(ValueError, match="query must not be empty"):
            reranker._score("", "some content")

    def test_score_empty_content_raises(self, reranker: DummyReranker):
        with pytest.raises(ValueError, match="content must not be empty"):
            reranker._score("some query", "")


class TestRerank:
    def test_rerank_orders_candidates_by_score(
        self,
        reranker: DummyReranker,
        hybrid_candidates,
    ):
        result = reranker.rerank(
            query="approve exception",
            candidates=hybrid_candidates,
            top_k=3,
        )

        scores = [item.rerank_score for item in result]

        assert scores == sorted(scores, reverse=True)

    def test_rerank_respects_top_k(
        self,
        reranker: DummyReranker,
        hybrid_candidates,
    ):
        result = reranker.rerank(
            query="approve exception",
            candidates=hybrid_candidates,
            top_k=2,
        )

        assert len(result) == 2

    def test_rerank_returns_all_when_top_k_is_larger(
        self,
        reranker: DummyReranker,
        hybrid_candidates,
    ):
        result = reranker.rerank(
            query="approve exception",
            candidates=hybrid_candidates,
            top_k=100,
        )

        assert len(result) == len(hybrid_candidates)

    def test_rerank_empty_candidates(
        self,
        reranker: DummyReranker,
    ):
        result = reranker.rerank(
            query="approve exception",
            candidates=[],
            top_k=5,
        )

        assert result == []

    def test_rerank_empty_query_raises(
        self,
        reranker: DummyReranker,
        hybrid_candidates,
    ):
        with pytest.raises(ValueError, match="query must not be empty"):
            reranker.rerank(
                query="",
                candidates=hybrid_candidates,
                top_k=5,
            )

    def test_rerank_invalid_top_k_raises(
        self,
        reranker: DummyReranker,
        hybrid_candidates,
    ):
        with pytest.raises(ValueError, match="top k must be greater than zero"):
            reranker.rerank(
                query="approve exception",
                candidates=hybrid_candidates,
                top_k=0,
            )

    def test_rerank_orders_candidates_by_score_match_id(
        self,
        reranker: DummyReranker,
        hybrid_candidates: list[HybridRetrievedChunk],
    ):
        result = reranker.rerank(
            query="Who can approve an exception to the annual leave policy?",
            candidates=hybrid_candidates,
            top_k=4,
        )

        assert [item.chunk.chunk.chunk_id for item in result] == [
            "chunk-b",
            "chunk-a",
            "chunk-c",
            "chunk-d",
        ]
