import json
from pathlib import Path

from rag_evaluation.chunking.text_chunker import TextChunker
from rag_evaluation.context.builder import ContextBuilder
from rag_evaluation.dataset.loader.text_document_loader import TextDocumentLoader
from rag_evaluation.embeddings.service import EmbeddingService
from rag_evaluation.evaluation.report import EvaluationReportBuilder
from rag_evaluation.evaluation.runner import EvaluationRunner
from rag_evaluation.judges.llm import LLMJudge
from rag_evaluation.models.cases import EvaluationCase
from rag_evaluation.models.chunks import Chunk, EmbeddedChunk
from rag_evaluation.models.reports import EvaluationCaseResult
from rag_evaluation.pipelines.rag_pipeline_v1 import RAGPipeline
from rag_evaluation.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_evaluation.providers.embedding.ollama_com import OllamaEmbeddingProvider
from rag_evaluation.providers.generation.ollama_com import OllamaLLMProvider
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


def evaluate_case(
    rag_pipeline: RAGPipeline,
    eval_runner: EvaluationRunner,
    case: EvaluationCase,
) -> EvaluationCaseResult:

    candidate_k = 10
    top_k = 5

    system_output = rag_pipeline.run(
        case.question, candidate_k=candidate_k, top_k=top_k
    )

    eval_case_result = eval_runner.evaluate(case, system_output)

    return eval_case_result


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

    # 9. create instance of retrieval pipline
    retrieval_pipeline = RetrievalPipeline(
        embedding_service=embedding_service,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    # 10. create instance of llm provider
    llm_provider = OllamaLLMProvider(model_name="mistral:7b-instruct-v0.3-q4_K_M")

    # 11. create instance of context builder
    context_builder = ContextBuilder()

    # 12. create instance of rag pipline
    rag_pipeline = RAGPipeline(retrieval_pipeline, context_builder, llm_provider)

    # 13. create instance of llm judge
    llm_judge = LLMJudge(llm_provider)

    # 14 create instance of Evaluation runner
    eval_runner = EvaluationRunner(llm_judge)

    results = []

    # debug_case = ["case-001", "case-002", "case-004", "case-005"]

    for case in cases:
        # if case.id not in debug_case:
        #     continue

        print(f"CASE: {case.id}")
        print(f"Question: {case.question}")
        print(f"Expected Answer: {case.expected_answer}")
        eval_case_result = evaluate_case(rag_pipeline, eval_runner, case)
        print(f"Generated Answer: {eval_case_result.system_output.generated_answer}")
        print(f"Diagnosis: {eval_case_result.diagnosis.value}\n\n")

        print("Metrics:")
        print(
            f"Recall={eval_case_result.retrieval_recall:.2f}, "
            f"Precision={eval_case_result.retrieval_precision:.2f}, "
            f"RR={eval_case_result.retrieval_rr:.2f}, "
            f"ContextRecall={eval_case_result.context_recall:.2f}, "
            f"ContextPrecision={eval_case_result.context_precision:.2f}, "
            f"Groundedness={eval_case_result.generation.groundedness.score:.2f}, "
            f"Correctness={eval_case_result.generation.correctness.score:.2f}, "
            f"Relevance={eval_case_result.generation.relevance.score:.2f}"
        )
        print("\nGroundedness Reason:")
        print(eval_case_result.generation.groundedness.reason)

        print("\nGroundedness Evidence:")
        print(eval_case_result.generation.groundedness.evidence)

        print("-" * 20, "\n\n")
        results.append(eval_case_result)

    report = EvaluationReportBuilder().build(results)

    print("\nRAG Evaluation Report")
    print("====================")
    print(f"Total cases:         {report.total_cases}")
    print(f"Retrieval Recall:    {report.retrieval_recall:.2f}")
    print(f"Retrieval Precision: {report.retrieval_precision:.2f}")
    print(f"Retrieval MRR:       {report.retrieval_mrr:.2f}")
    print(f"Context Recall:      {report.context_recall:.2f}")
    print(f"Context Precision:   {report.context_precision:.2f}")
    print(f"Groundedness:        {report.groundedness:.2f}")
    print(f"Correctness:         {report.correctness:.2f}")
    print(f"Relevance:           {report.relevance:.2f}")
    print(f"Avg Latency:         {report.avg_latency_ms:.2f} ms")
    print(f"P95 Latency:         {report.p95_latency_ms:.2f} ms")
    print(f"Avg Input Tokens:    {report.avg_input_tokens:.2f}")
    print(f"Avg Output Tokens:   {report.avg_output_tokens:.2f}")
    print(f"Avg Cost:            ${report.avg_cost_usd:.4f}")


if __name__ == "__main__":
    main()
