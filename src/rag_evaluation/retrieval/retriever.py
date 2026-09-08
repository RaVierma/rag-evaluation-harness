from rag_evaluation.models import EmbeddedChunk
from rag_evaluation.models import RetrievedChunk
from rag_evaluation.retrieval.similarity import cosine_similarity


class InMemoryRetriever:
    def __init__(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ):
        self._embedded_chunks = embedded_chunks

    def retrieve(
        self,
        query_embedding: tuple[float, ...],
        top_k: int,
    ) -> list[RetrievedChunk]:

        if not query_embedding:
            raise ValueError("Not valid query embedding")

        if top_k <= 0:
            raise ValueError("tok k must be greater that zero.")

        if self._embedded_chunks:
            expected_dimension = len(self._embedded_chunks[0].embedding)

            if len(query_embedding) != expected_dimension:
                raise ValueError("embedding query dimension mismatch.")

        retrieve_chunks: list[RetrievedChunk] = []

        for embed_chunk in self._embedded_chunks:
            score = cosine_similarity(query_embedding, embed_chunk.embedding)
            retrieve_chunks.append(RetrievedChunk(chunk=embed_chunk, score=score))

        return sorted(retrieve_chunks, key=lambda x: x.score, reverse=True)[:top_k]
