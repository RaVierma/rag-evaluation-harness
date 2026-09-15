from rag.embeddings.similarity import cosine_similarity
from rag.models.chunks import EmbeddedChunk, RetrievedChunk
from rag.retriever import Retriever


class InMemoryRetriever(Retriever):
    def __init__(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ):
        self._embedded_chunks = embedded_chunks

    def retrieve(
        self,
        query_embedding: tuple[float, ...],
        candidate_k: int,
    ) -> list[RetrievedChunk]:

        if not query_embedding:
            raise ValueError("Not valid query embedding")

        if candidate_k <= 0:
            raise ValueError("tok k must be greater that zero.")

        if self._embedded_chunks:
            expected_dimension = len(self._embedded_chunks[0].embedding)

            if len(query_embedding) != expected_dimension:
                raise ValueError("embedding query dimension mismatch.")

        retrieve_chunks: list[RetrievedChunk] = []

        for embed_chunk in self._embedded_chunks:
            score = cosine_similarity(query_embedding, embed_chunk.embedding)
            retrieve_chunks.append(RetrievedChunk(chunk=embed_chunk, score=score))

        return sorted(retrieve_chunks, key=lambda x: x.score, reverse=True)[
            :candidate_k
        ]
