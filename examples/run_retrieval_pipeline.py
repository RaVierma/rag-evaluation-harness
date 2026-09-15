from pathlib import Path

from rag.chunking import TextChunker
from rag.embeddings import EmbeddingService
from rag.ingestion import TextDocumentLoader
from rag.models.chunks import Chunk, EmbeddedChunk
from rag.models.pipeline import RetrievalOutput
from rag.pipelines import RetrievalPipeline
from rag.providers.embedding import OllamaEmbeddingProvider
from rag.reranking import CrossEncoderReranker
from rag.retriever import BM25Retriever, HybridRetriever, InMemoryRetriever

DOCUMENTS_PATH = Path("dataset/documents")

QUERIES = [
    "How many vacation days can employees carry forward?",
    "What happens to unused vacation days above the carryover limit?",
    "Can an employee get an exception to the vacation carryover limit?",
    "What are the requirements for working remotely?",
    "Who can approve an exception to the annual leave policy?",
]


def chunk_documents() -> list[Chunk]:
    loader = TextDocumentLoader(DOCUMENTS_PATH)
    documents = loader.load()

    chunker = TextChunker(
        chunk_size=300,
        overlap=50,
    )

    chunks: list[Chunk] = []

    for document in documents:
        chunks.extend(chunker.chunk(document))

    return chunks


def embed_chunks(
    chunks: list[Chunk],
    embedding_service: EmbeddingService,
) -> list[EmbeddedChunk]:

    return embedding_service.embed_chunks(chunks)


def run_retrieval_pipeline_experiment(
    embedding_service: EmbeddingService,
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    dense_retriever = InMemoryRetriever(embedded_chunks)
    bm25_retriever = BM25Retriever(embedded_chunks)

    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)

    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    for query in QUERIES:
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        retrieval_pipeline = RetrievalPipeline(
            embedding_service, hybrid_retriever, reranker
        )

        pipeline_result: RetrievalOutput = retrieval_pipeline.retrieve(
            query, candidate_k=20, top_k=5
        )

        for rank, result in enumerate(pipeline_result.results, start=1):
            print(f"\nRank: {rank}")
            print(f"Chunk: {result.chunk.chunk.chunk_id}")
            print(f"Dense score: {(result.chunk.dense_score or 0.0):.4f}")
            print(f"BM25 score: {(result.chunk.bm25_score or 0.0):.4f}")
            print(f"Dense rank: {result.chunk.dense_rank}")
            print(f"BM25 rank: {result.chunk.bm25_rank}")
            print(f"RRF score: {result.chunk.rrf_score:.4f}")
            print(f"ReRank score: {result.rerank_score:.4f}")
            print(f"Content:\n{result.chunk.chunk.content}")


def main() -> None:
    # 1. Chunk documents
    chunks = chunk_documents()

    print(f"Documents chunked: {len(chunks)}")

    # 2. Dummy embedding provider
    embedding_provider = OllamaEmbeddingProvider(
        model_name="nomic-embed-text:latest",
        dimension=768,
    )

    embedding_service = EmbeddingService(embedding_provider)

    # 3. Embed document chunks
    embedded_chunks = embed_chunks(
        chunks,
        embedding_service,
    )

    print(f"Chunks embedded: {len(embedded_chunks)}")

    # 4. Run retrieval experiment
    run_retrieval_pipeline_experiment(
        embedding_service=embedding_service,
        embedded_chunks=embedded_chunks,
    )


if __name__ == "__main__":
    main()
