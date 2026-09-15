from rag.models.chunks import HybridRetrievedChunk
from rag.retriever import BM25Retriever, Retriever


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        self._dense_retriever = dense_retriever
        self._bm25_retriever = bm25_retriever

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        query_embedding: tuple[float, ...],
        candidate_k: int,
        pool_k: int,
    ) -> list[HybridRetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if not query_embedding:
            raise ValueError("query embedding must not be empty")

        if pool_k <= 0 or candidate_k <= 0:
            raise ValueError("pool_k and candidate_k must be greater than 0")

        dense_retrieve = {
            retrieve_chunk.chunk.chunk_id: (retrieve_chunk, rank)
            for rank, retrieve_chunk in enumerate(
                self._dense_retriever.retrieve(query_embedding, candidate_k), 1
            )
        }

        bm25_retrieve = {
            retrieve_chunk.chunk.chunk_id: (retrieve_chunk, rank)
            for rank, retrieve_chunk in enumerate(
                self._bm25_retriever.retrieve(query, candidate_k), 1
            )
        }

        hybrid_retrieve: list[HybridRetrievedChunk] = []

        unique_chunk_ids = set(dense_retrieve.keys()).union(set(bm25_retrieve.keys()))

        for chunk_id in unique_chunk_ids:
            drchunk_with_drank = dense_retrieve.get(chunk_id)

            bm25_rchunk_with_rank = bm25_retrieve.get(chunk_id)

            dense_score = None
            bm25_score = None
            dense_rank = None
            bm25_rank = None
            rrf_score = 0.0
            chunk = None

            if drchunk_with_drank:
                dense_score = drchunk_with_drank[0].score
                dense_rank = drchunk_with_drank[1]
                chunk = drchunk_with_drank[0].chunk
                rrf_score = 1 / (self._rrf_k + drchunk_with_drank[1]) + rrf_score

            if bm25_rchunk_with_rank:
                bm25_score = bm25_rchunk_with_rank[0].score
                bm25_rank = bm25_rchunk_with_rank[1]
                chunk = bm25_rchunk_with_rank[0].chunk
                rrf_score = 1 / (self._rrf_k + bm25_rchunk_with_rank[1]) + rrf_score

            hybrid_retrieve.append(
                HybridRetrievedChunk(
                    chunk=chunk,
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    dense_rank=dense_rank,
                    bm25_rank=bm25_rank,
                    rrf_score=rrf_score,
                )
            )

        return sorted(
            hybrid_retrieve,
            key=lambda item: (
                -item.rrf_score,
                item.chunk.chunk_id,
            ),
        )[:pool_k]
