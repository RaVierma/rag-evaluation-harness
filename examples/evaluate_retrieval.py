import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_evaluation.chunking.text_chunker import TextChunker
from rag_evaluation.dataset.loader.text_document_loader import TextDocumentLoader
from rag_evaluation.embeddings.providers.ollama_com import OllamaEmbeddingProvider
from rag_evaluation.embeddings.service import EmbeddingService
from rag_evaluation.evaluation.retrieval import evaluate_retrieval
from rag_evaluation.models.cases import EvaluationCase
from rag_evaluation.models.chunks import Chunk, EmbeddedChunk
from rag_evaluation.pipeline.retrieval_pipeline import RetrievalPipeline
from rag_evaluation.reranking.cross_encoder import CrossEncoderReranker
from rag_evaluation.retriever.bm25 import BM25Retriever
from rag_evaluation.retriever.hybrid import HybridRetriever
from rag_evaluation.retriever.in_memory import InMemoryRetriever

DOCUMENTS_PATH = Path("dataset/documents")
GOLDEN_DATASET_PATH = Path("dataset/golden.jsonl")


def load_cases(path: Path) -> list[EvaluationCase]:

    if not path.is_file():
        raise FileNotFoundError(f"{path} not exists.")

    is_jsonl = str(path).endswith(".jsonl")

    if not is_jsonl:
        raise FileNotFoundError("must be jsonl.")

    return [
        EvaluationCase(**json.loads(line))
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
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


def check(model):
    query = "Who may request an exception?"

    candidate_a = """If an employee was unable to use accrued leave because of documented business requirements, the manager may request an exception to the standard five-day carryover limit.

    Approved exceptions must be documented and submitted to HR."""

    candidate_b = """Unused vacation days above the five-day carryover limit normally expire at the end of the calendar year.

    Exceptions may be granted when business circumstances prevented the employee from taking approved leave, subject to manager and HR approval."""

    scores = model.predict(
        [
            (query, candidate_a),
            (query, candidate_b),
        ]
    )

    print(scores)


def evaluate_case(
    pipeline: RetrievalPipeline,
    case: EvaluationCase,
) -> dict[str, Any]:

    candidate_k = 10
    top_k = 5

    retrieval_output = pipeline.retrieve(
        case.question, candidate_k=candidate_k, top_k=top_k
    )
    chunk_id_to_relevance_judge = {
        cs.chunk_id: cs.relevance for cs in case.relevance_judgments
    }

    metrics = evaluate_retrieval(retrieval_output, case, k=top_k)

    print(f"Recall@{top_k}: {metrics['recall_at_k']}")
    print(f"RR@{top_k}: {metrics['rr_at_k']}")
    print(f"NDCG@{top_k}: {metrics['ndcg_at_k']}\n\n")

    for rank, rrchunk in enumerate(retrieval_output.results, start=1):
        # relevance_map = {rel.chunk_id: rel.relevance for rel in case.relevance_judgments}
        # print(
        #     f"{rank}. {rrchunk.chunk.chunk.chunk_id}"
        #     f" | rerank={rrchunk.rerank_score:.4f}"
        #     f" | relevance={relevance_map.get(rrchunk.chunk.chunk.chunk_id, 0)}"
        # )

        # print(f"   {rrchunk.chunk.chunk.content}")
        # continue
        chunk_id = rrchunk.chunk.chunk.chunk_id
        relevance = chunk_id_to_relevance_judge.get(chunk_id, 0)

        print("Ranked results:")
        print(f"{rank}. {chunk_id} -> relevance={relevance}")

    return metrics


def main() -> None:
    # 1. Chunk documents
    chunks = chunk_documents()

    print(f"Documents chunked: {len(chunks)}")

    # 2. load cases
    cases = load_cases(GOLDEN_DATASET_PATH)

    print(f"Cases: {len(cases)}")

    # 3. embedding provider
    embedding_provider = OllamaEmbeddingProvider(
        model_name="nomic-embed-text:latest",
        dimension=768,
    )

    # 4. embedding service
    embedding_service = EmbeddingService(embedding_provider)

    # 5. embedding the chunks
    embedded_chunks: list[EmbeddedChunk] = embedding_service.embed_chunks(chunks)

    # 6. initalize dense, bm25 retriever
    dense_retriever = InMemoryRetriever(embedded_chunks)
    bm25_retriever = BM25Retriever(embedded_chunks)

    # 7. initalizer hybrid retriever with dense and bm25
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)

    # 8. create instance of cross encoder reranker
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # check(reranker._model)
    # return

    # 9. instancitate retrieval pipline
    pipeline = RetrievalPipeline(
        embedding_service=embedding_service,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    results = []

    for case in cases:
        print(f"CASE: {case.id}")
        print(f"Question: {case.question}\n\n")
        metrics = evaluate_case(pipeline, case)

        recall = metrics["recall_at_k"]
        rr = metrics["rr_at_k"]
        ndcg = metrics["ndcg_at_k"]

        results.append(
            {
                "case_id": case.id,
                "category": case.category.value,
                "recall": recall,
                "rr": rr,
                "ndcg": ndcg,
            }
        )
        print("\n")

    average_recall = sum(result["recall"] for result in results) / len(results)

    average_rr = sum(result["rr"] for result in results) / len(results)

    average_ndcg = sum(result["ndcg"] for result in results) / len(results)

    category_results = defaultdict(list)

    for result in results:
        category_results[result["category"]].append(result)

    print("\n================ RETRIEVAL EVALUATION ================")
    print(f"Cases: {len(results)}")
    print("Candidate K: 20")
    print("Top K: 5")
    print(f"Average Recall@5: {average_recall:.3f}")
    print(f"Average RR@5:     {average_rr:.3f}")
    print(f"Average NDCG@5:   {average_ndcg:.3f}")
    print("=======================================================")

    print("\nCATEGORY PERFORMANCE")
    print("--------------------")

    for category, category_cases in category_results.items():
        avg_recall = sum(result["recall"] for result in category_cases) / len(
            category_cases
        )

        avg_rr = sum(result["rr"] for result in category_cases) / len(category_cases)

        avg_ndcg = sum(result["ndcg"] for result in category_cases) / len(
            category_cases
        )

        print(
            f"{category}: "
            f"cases={len(category_cases)}, "
            f"Recall@5={avg_recall:.3f}, "
            f"RR@5={avg_rr:.3f}, "
            f"NDCG@5={avg_ndcg:.3f}"
        )

    worst_cases = sorted(
        results,
        key=lambda result: result["ndcg"],
    )

    print("\nWORST CASES")
    print("-----------")

    for result in worst_cases[:3]:
        print(f"{result['case_id']} → NDCG@5={result['ndcg']:.3f}")

    # Find failures
    failures = [result for result in results if result["ndcg"] < 1.0]

    print("\nFAILURES")
    print("--------")

    for failure in failures:
        print(f"{failure['case_id']} → NDCG@5={failure['ndcg']:.3f}")


if __name__ == "__main__":
    main()
