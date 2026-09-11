from rag_evaluation.embeddings import DummyEmbeddingProvider, EmbeddingService
from rag_evaluation.models.chunks import Chunk

embedding_provider = DummyEmbeddingProvider()


def test_embedding_service_for_embedding_and_it_ordered():
    chunks = [
        Chunk.model_construct(
            id="doc-000-chunk-000",
            document_id="doc-000",
            content="Hello world",
        ),
        Chunk.model_construct(
            id="doc-000-chunk-001",
            document_id="doc-000",
            content="I'm here for you.",
        ),
        Chunk.model_construct(
            id="doc-000-chunk-002",
            document_id="doc-000",
            content="are you there.",
        ),
    ]

    service = EmbeddingService(embedding_provider)

    embedded = service.embed_chunks(chunks)

    assert embedded[0].chunk_id == chunks[0].id
    assert embedded[1].chunk_id == chunks[1].id
    assert embedded[2].chunk_id == chunks[2].id

    assert embedded[0].embedding == embedding_provider.embed(chunks[0].content)
