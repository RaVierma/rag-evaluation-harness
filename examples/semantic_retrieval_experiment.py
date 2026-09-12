from pathlib import Path

from rag_evaluation.chunking import TextChunker
from rag_evaluation.dataset import TextDocumentLoader
from rag_evaluation.embeddings.service import EmbeddingService
from rag_evaluation.models.chunks import Chunk, EmbeddedChunk
from rag_evaluation.providers.embedding.ollama_com import OllamaEmbeddingProvider
from rag_evaluation.retriever import InMemoryRetriever

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
    embedding_provider: OllamaEmbeddingProvider,
) -> list[EmbeddedChunk]:
    embedding_service = EmbeddingService(embedding_provider)

    return embedding_service.embed_chunks(chunks)


def run_retrieval_experiment(
    provider: OllamaEmbeddingProvider,
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    retriever = InMemoryRetriever(embedded_chunks)

    for query in QUERIES:
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        query_embedding = provider.embed(query)

        results = retriever.retrieve(
            query_embedding=query_embedding,
            candidate_k=5,
        )

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Chunk: {result.chunk.chunk_id}")
            print(f"Score: {result.score:.4f}")
            print(f"Content:\n{result.chunk.content}")


def main() -> None:
    # 1. Chunk documents
    chunks = chunk_documents()

    print(f"Documents chunked: {len(chunks)}")

    # 2. Real semantic embedding provider
    embedding_provider = OllamaEmbeddingProvider(
        model_name="nomic-embed-text:latest",
        dimension=768,
    )

    # 3. Embed document chunks
    embedded_chunks = embed_chunks(
        chunks,
        embedding_provider,
    )

    print(f"Chunks embedded: {len(embedded_chunks)}")

    # 4. Run retrieval experiment
    run_retrieval_experiment(
        provider=embedding_provider,
        embedded_chunks=embedded_chunks,
    )


if __name__ == "__main__":
    main()
