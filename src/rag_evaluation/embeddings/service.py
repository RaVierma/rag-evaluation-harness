from rag_evaluation.embeddings.base import EmbeddingProvider
from rag_evaluation.models import Chunk, EmbeddedChunk


class EmbeddingService:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self._embedding_provider = embedding_provider

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embedding = self._embedding_provider.embed_batch(texts)

        return [
            EmbeddedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embedding)
        ]
