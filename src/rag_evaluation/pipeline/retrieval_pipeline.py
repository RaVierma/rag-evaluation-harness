from rag_evaluation.embeddings.service import EmbeddingService
from rag_evaluation.models.chunks import RetrievalOutput
from rag_evaluation.reranking.base import Reranker
from rag_evaluation.retriever import HybridRetriever


class RetrievalPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
    ):
        self.embedding_service = embedding_service
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 5,
    ) -> RetrievalOutput:
        if not query.strip():
            raise ValueError("query must not be empty")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if top_k > candidate_k:
            raise ValueError("top_k cannot be greater than candidate_k")

        query_embedding = self.embedding_service.embed_query(query)

        candidates = self.hybrid_retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            candidate_k=candidate_k,
            pool_k=candidate_k,
        )

        rerank_chunk = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

        return RetrievalOutput(
            query=query, results=rerank_chunk, candidate_k=candidate_k, top_k=top_k
        )
