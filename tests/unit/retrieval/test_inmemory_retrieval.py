import pytest

from rag_evaluation.models.chunks import EmbeddedChunk
from rag_evaluation.retrieval.retriever import InMemoryRetriever


def test_inmemory_retrieval_for_valid_data():
    chunks = [
        EmbeddedChunk(
            chunk_id="01",
            document_id="doc-1",
            content="How",
            embedding=(1.0, 0.0),
        ),
        EmbeddedChunk(
            chunk_id="02",
            document_id="doc-1",
            content="Are",
            embedding=(0.0, 1.0),
        ),
        EmbeddedChunk(
            chunk_id="03",
            document_id="doc-1",
            content="You",
            embedding=(0.9, 0.1),
        ),
    ]

    query_embedding = (1.0, 0.0)

    retriever = InMemoryRetriever(chunks)

    retrieval_result = retriever.retrieve(query_embedding, top_k=2)

    assert len(retrieval_result) == 2

    assert retrieval_result[0].chunk.chunk_id == "01"

    assert retrieval_result[0].score > retrieval_result[1].score
