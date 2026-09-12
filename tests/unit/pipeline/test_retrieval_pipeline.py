from unittest.mock import Mock

import pytest

from rag_evaluation.models.chunks import (
    EmbeddedChunk,
    HybridRetrievedChunk,
    RerankedChunk,
)
from rag_evaluation.pipelines.retrieval_pipeline import RetrievalPipeline


def make_reranked_chunk(chunk_id: str) -> RerankedChunk:
    embedded_chunk = EmbeddedChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        content=f"content for {chunk_id}",
        embedding=(0.1, 0.2, 0.3),
    )

    hybrid_chunk = HybridRetrievedChunk(
        chunk=embedded_chunk,
        dense_score=0.9,
        bm25_score=0.8,
        dense_rank=1,
        bm25_rank=1,
        rrf_score=0.03,
    )

    return RerankedChunk(
        chunk=hybrid_chunk,
        rerank_score=5.0,
    )


@pytest.fixture
def embedding_service():
    service = Mock()
    service.embed_query.return_value = (0.1, 0.2, 0.3)
    return service


@pytest.fixture
def hybrid_retriever():
    retriever = Mock()
    retriever.retrieve.return_value = [
        make_reranked_chunk("chunk-001"),
        make_reranked_chunk("chunk-002"),
    ]
    return retriever


@pytest.fixture
def reranker():
    reranker = Mock()

    reranker.rerank.return_value = [
        make_reranked_chunk("chunk-001"),
    ]

    return reranker


@pytest.fixture
def pipeline(embedding_service, hybrid_retriever, reranker):
    return RetrievalPipeline(
        embedding_service=embedding_service,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )


def test_retrieve_returns_reranked_results(
    pipeline,
    embedding_service,
    hybrid_retriever,
    reranker,
):
    output = pipeline.retrieve(
        query="Who can approve an exception?",
        candidate_k=20,
        top_k=5,
    )

    results = output.results
    assert output.candidate_k == 20
    assert output.top_k == 5

    assert len(results) == 1
    assert results[0].chunk.chunk.chunk_id == "chunk-001"

    embedding_service.embed_query.assert_called_once_with(
        "Who can approve an exception?"
    )

    hybrid_retriever.retrieve.assert_called_once_with(
        query="Who can approve an exception?",
        query_embedding=(0.1, 0.2, 0.3),
        candidate_k=20,
        pool_k=20,
    )

    reranker.rerank.assert_called_once_with(
        query="Who can approve an exception?",
        candidates=hybrid_retriever.retrieve.return_value,
        top_k=5,
    )


def test_retrieve_passes_candidate_k_and_top_k(
    pipeline,
    embedding_service,
    hybrid_retriever,
    reranker,
):
    pipeline.retrieve(
        query="test query",
        candidate_k=50,
        top_k=10,
    )

    hybrid_retriever.retrieve.assert_called_once_with(
        query="test query",
        query_embedding=(0.1, 0.2, 0.3),
        candidate_k=50,
        pool_k=50,
    )

    reranker.rerank.assert_called_once_with(
        query="test query",
        candidates=hybrid_retriever.retrieve.return_value,
        top_k=10,
    )


def test_retrieve_rejects_empty_query(pipeline):
    with pytest.raises(ValueError, match="query must not be empty"):
        pipeline.retrieve("")


def test_retrieve_rejects_whitespace_query(pipeline):
    with pytest.raises(ValueError, match="query must not be empty"):
        pipeline.retrieve("   ")


def test_retrieve_rejects_invalid_candidate_k(pipeline):
    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than zero",
    ):
        pipeline.retrieve(
            query="test query",
            candidate_k=0,
            top_k=5,
        )


def test_retrieve_rejects_invalid_top_k(pipeline):
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        pipeline.retrieve(
            query="test query",
            candidate_k=20,
            top_k=0,
        )


def test_retrieve_rejects_top_k_greater_than_candidate_k(pipeline):
    with pytest.raises(
        ValueError,
        match="top_k cannot be greater than candidate_k",
    ):
        pipeline.retrieve(
            query="test query",
            candidate_k=5,
            top_k=10,
        )


def test_retrieve_does_not_call_dependencies_when_validation_fails(
    pipeline,
    embedding_service,
    hybrid_retriever,
    reranker,
):
    with pytest.raises(ValueError):
        pipeline.retrieve(
            query="",
            candidate_k=20,
            top_k=5,
        )

    embedding_service.embed_query.assert_not_called()
    hybrid_retriever.retrieve.assert_not_called()
    reranker.rerank.assert_not_called()
