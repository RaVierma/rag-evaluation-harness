from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tqdm import tqdm

from rag_evaluation.chunking import TextChunker
from rag_evaluation.context.builder import ContextBuilder
from rag_evaluation.dataset.loader.text_document_loader import TextDocumentLoader
from rag_evaluation.embeddings import EmbeddingService
from rag_evaluation.evaluation import (
    EvaluationRunner,
    EvaluationWorkflow,
    EvaluationComparisonBuilder,
    EvaluationDecisionBuilder,
    RegressionDecisionBuilder,
)
from rag_evaluation.evaluation import ReleaseStatus, determine_release_status
from rag_evaluation.judges import LLMJudge
from rag_evaluation.models import (
    Chunk,
    EmbeddedChunk,
    EvaluationCase,
    RAGPipelineResult,
    EvaluationDecision,
    EvaluationThresholds,
    RegressionDecision,
    RegressionThresholds,
    EvaluationReport,
)
from rag_evaluation.pipelines import RAGPipeline, RetrievalPipeline
from rag_evaluation.providers.embedding import OllamaEmbeddingProvider
from rag_evaluation.providers.generation import OllamaLLMProvider
from rag_evaluation.reranking.cross_encoder import CrossEncoderReranker
from rag_evaluation.retriever import (
    BM25Retriever,
    HybridRetriever,
    InMemoryRetriever,
)


# Evaluation configuration
@dataclass(frozen=True)
class EvaluationConfig:
    name: str
    candidate_k: int
    top_k: int


V1_CONFIG = EvaluationConfig(
    name="v1",
    candidate_k=10,
    top_k=5,
)

V2_CONFIG = EvaluationConfig(
    name="v2",
    candidate_k=10,
    top_k=3,
)


# Dataset / model configuration
DOCUMENTS_PATH = Path("dataset/documents")
GOLDEN_DATASET_PATH = Path("dataset/golden.jsonl")

LLM_MODEL = "mistral:7b-instruct-v0.3-q4_K_M"
EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_DIMENSION = 768
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# Dataset preparation
def load_documents() -> list[Chunk]:
    loader = TextDocumentLoader(DOCUMENTS_PATH)
    chunker = TextChunker(chunk_size=300, overlap=50)

    documents = loader.load()

    chunks: list[Chunk] = []
    for document in documents:
        document_chunks = chunker.chunk(document)

        chunks.extend(document_chunks)

    return chunks


def load_cases() -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []

    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                cases.append(EvaluationCase.model_validate_json(line))

    return cases


# Quality Thresholds
def build_quality_thresholds() -> EvaluationThresholds:
    return EvaluationThresholds(
        min_retrieval_recall=0.90,
        min_retrieval_precision=0.70,
        min_retrieval_mrr=0.90,
        min_context_precision=0.70,
        min_groundedness=0.90,
        min_correctness=0.90,
        max_p95_latency_ms=30000.0,
        max_avg_cost_usd=1.00,
    )


# Regression Thresholds
def build_regression_thresholds() -> RegressionThresholds:
    return RegressionThresholds(
        min_retrieval_recall_drop=0.03,
        min_retrieval_precision_drop=0.03,
        min_context_recall_drop=0.03,
        min_context_precision_drop=0.03,
        min_groundedness_drop=0.03,
        min_correctness_drop=0.03,
        min_relevance_drop=0.03,
        max_p95_latency_ms_up=5000.0,
        max_avg_cost_usd_up=0.05,
    )


# Embedding / retrieval
def build_embedding_service() -> EmbeddingService:
    provider = OllamaEmbeddingProvider(
        model_name=EMBEDDING_MODEL, dimension=EMBEDDING_DIMENSION
    )

    return EmbeddingService(
        embedding_provider=provider,
    )


def build_retrieval_pipeline(
    embedding_service: EmbeddingService,
    embedded_chunks: list[EmbeddedChunk],
) -> RetrievalPipeline:
    vector_retriever = InMemoryRetriever(
        embedded_chunks=embedded_chunks,
    )

    bm25_retriever = BM25Retriever(
        documents=embedded_chunks,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    reranker = CrossEncoderReranker(
        model_name=RERANKER_MODEL,
    )

    return RetrievalPipeline(
        embedding_service=embedding_service,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )


# LLM / RAG pipeline
def build_llm_provider() -> OllamaLLMProvider:
    return OllamaLLMProvider(
        model_name=LLM_MODEL,
    )


def build_rag_pipeline(
    retrieval_pipeline: RetrievalPipeline,
    llm_provider: OllamaLLMProvider,
) -> RAGPipeline:
    context_builder = ContextBuilder()

    return RAGPipeline(
        retrieval_pipeline=retrieval_pipeline,
        context_builder=context_builder,
        llm_provider=llm_provider,
    )


# Evaluation workflow
def build_evaluation_workflow(
    llm_provider: OllamaLLMProvider,
) -> EvaluationWorkflow:
    judge = LLMJudge(
        provider=llm_provider,
    )

    evaluation_runner = EvaluationRunner(
        judge=judge,
    )

    decision_builder = EvaluationDecisionBuilder()

    comparison_builder = EvaluationComparisonBuilder()

    regression_builder = RegressionDecisionBuilder()

    return EvaluationWorkflow(
        evaluation_runner=evaluation_runner,
        decision_builder=decision_builder,
        comparison_builder=comparison_builder,
        regression_builder=regression_builder,
    )


# RAG execution
def run_rag_pipeline(
    rag_pipeline: RAGPipeline,
    cases: list[EvaluationCase],
    config: EvaluationConfig,
) -> list[RAGPipelineResult]:
    results: list[RAGPipelineResult] = []

    for case in tqdm(cases, desc="RAG Pipline", total=len(cases)):
        result = rag_pipeline.run(
            query=case.question,
            candidate_k=config.candidate_k,
            top_k=config.top_k,
        )

        results.append(result)

    return results


# Evaluation execution
def run_evaluation(
    rag_pipeline: RAGPipeline,
    workflow: EvaluationWorkflow,
    cases: list[EvaluationCase],
    config: EvaluationConfig,
):
    print()
    print("=" * 70)
    print(f"Running {config.name.upper()}")
    print(f"candidate_k = {config.candidate_k}")
    print(f"top_k       = {config.top_k}")
    print("=" * 70)

    pipeline_results = run_rag_pipeline(
        rag_pipeline=rag_pipeline,
        cases=cases,
        config=config,
    )

    report = workflow.evaluate(
        cases=cases,
        pipeline_results=pipeline_results,
    )

    return report


# Evaluation for Quality
def evaluate_quality_gate(
    workflow: EvaluationWorkflow,
    report: EvaluationReport,
) -> EvaluationDecision:
    thresholds = build_quality_thresholds()

    return workflow.quality_gate(
        report=report,
        thresholds=thresholds,
    )


# Evaluation Comparision
def compare_evaluations(
    workflow: EvaluationWorkflow,
    v1_report: EvaluationReport,
    v2_report: EvaluationReport,
):
    return workflow.compare(
        v1_report=v1_report,
        v2_report=v2_report,
        v1_decision=None,
        v2_decision=None,
    )


# Reporting
def print_report(
    name: str,
    report,
) -> None:
    print()
    print("-" * 70)
    print(f"{name} Evaluation Report")
    print("-" * 70)

    print(f"Total Cases:            {report.total_cases}")

    print()
    print("Retrieval")
    print(f"  Recall:               {report.retrieval_recall:.2f}")
    print(f"  Precision:            {report.retrieval_precision:.2f}")
    print(f"  MRR:                  {report.retrieval_mrr:.2f}")

    print()
    print("Context")
    print(f"  Recall:               {report.context_recall:.2f}")
    print(f"  Precision:            {report.context_precision:.2f}")

    print()
    print("Generation")
    print(f"  Groundedness:         {report.groundedness:.2f}")
    print(f"  Correctness:          {report.correctness:.2f}")
    print(f"  Relevance:            {report.relevance:.2f}")

    print()
    print("Performance")
    print(f"  Avg Latency:          {report.avg_latency_ms:.2f} ms")
    print(f"  P95 Latency:          {report.p95_latency_ms:.2f} ms")
    print(f"  Avg Input Tokens:     {report.avg_input_tokens:.2f}")
    print(f"  Avg Output Tokens:    {report.avg_output_tokens:.2f}")
    print(f"  Avg Cost:             ${report.avg_cost_usd:.4f}")

    print("-" * 70)


# Comparing
def print_comparison(comparison) -> None:
    print()
    print("=" * 70)
    print("V1 → V2 Comparison")
    print("=" * 70)

    for metric in comparison.metric_comparisons:
        percentage = (
            f"{metric.percentage_delta:+.2f}%"
            if metric.percentage_delta is not None
            else "N/A"
        )

        print(
            f"{metric.name:<25} "
            f"V1={metric.v1_value:.4f} "
            f"V2={metric.v2_value:.4f} "
            f"Δ={metric.absolute_delta:+.4f} "
            f"({percentage})"
        )

    print()
    print(f"V1 Quality Gate: {'PASS' if comparison.v1_decision.is_passed else 'FAIL'}")
    print(f"V2 Quality Gate: {'PASS' if comparison.v2_decision.is_passed else 'FAIL'}")
    print("=" * 70)


def print_regression_decision(
    decision: RegressionDecision,
) -> None:
    print()
    print("=" * 70)
    print("Regression Gate")
    print("=" * 70)

    print(f"Overall: {'PASS' if decision.is_passed else 'FAIL'}")

    if decision.passed:
        print()
        print("Passed:")
        for metric in decision.passed:
            print(
                f"  ✓ {metric.name}: "
                f"V1={metric.v1_value:.4f}, "
                f"V2={metric.v2_value:.4f}, "
                f"Δ={metric.actual_delta:+.4f}, "
                f"threshold={metric.threshold_delta:.4f}"
            )

    if decision.failures:
        print()
        print("Failures:")
        for metric in decision.failures:
            print(
                f"  ✗ {metric.name}: "
                f"V1={metric.v1_value:.4f}, "
                f"V2={metric.v2_value:.4f}, "
                f"Δ={metric.actual_delta:+.4f}, "
                f"threshold={metric.threshold_delta:.4f}"
            )

    print("=" * 70)


# Realise Decision
def print_release_decision(
    status: ReleaseStatus,
) -> None:
    print()
    print("=" * 70)
    print("Release Decision")
    print("=" * 70)

    if status == ReleaseStatus.RELEASE:
        print("Status: RELEASE")
        print("V2 passed the quality and regression gates.")

    elif status == ReleaseStatus.REVIEW:
        print("Status: REVIEW")
        print("V2 requires engineering review before release.")

    else:
        print("Status: REJECT")
        print("V2 failed the release criteria.")

    print("=" * 70)


def print_release_report(
    v1_report: EvaluationReport,
    v2_report: EvaluationReport,
    comparison,
    v1_decision: EvaluationDecision,
    v2_decision: EvaluationDecision,
    regression_decision: RegressionDecision,
    release_status: ReleaseStatus,
) -> None:
    print()
    print("=" * 70)
    print("RAG RELEASE EVALUATION REPORT")
    print("=" * 70)

    print()
    print("Baseline")
    print("  V1: candidate_k=10, top_k=5")

    print()
    print("Candidate")
    print("  V2: candidate_k=10, top_k=3")

    print()
    print("Quality Gates")
    print(f"  V1: {'PASS' if v1_decision.is_passed else 'FAIL'}")
    print(f"  V2: {'PASS' if v2_decision.is_passed else 'FAIL'}")

    print()
    print("Key Improvements")

    improvements = [
        "retrieval_precision",
        "context_precision",
        "avg_latency_ms",
        "p95_latency_ms",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_cost_usd",
    ]

    metric_map = {metric.name: metric for metric in comparison.metric_comparisons}

    for metric_name in improvements:
        metric = metric_map.get(metric_name)

        if metric is None:
            continue

        if metric.absolute_delta > 0:
            direction = "↑"
        elif metric.absolute_delta < 0:
            direction = "↓"
        else:
            direction = "→"

        percentage = (
            f"{metric.percentage_delta:+.2f}%"
            if metric.percentage_delta is not None
            else "N/A"
        )

        print(
            f"  {metric.name}: "
            f"{metric.v1_value:.4f} → "
            f"{metric.v2_value:.4f} "
            f"{direction} ({percentage})"
        )

    print()
    print("Regressions")

    regressions = [
        metric for metric in comparison.metric_comparisons if metric.absolute_delta < 0
    ]

    if not regressions:
        print("  None")

    for metric in regressions:
        percentage = (
            f"{metric.percentage_delta:+.2f}%"
            if metric.percentage_delta is not None
            else "N/A"
        )

        print(
            f"  {metric.name}: "
            f"{metric.v1_value:.4f} → "
            f"{metric.v2_value:.4f} "
            f"({percentage})"
        )

    print()
    print(f"Regression Gate: {'PASS' if regression_decision.is_passed else 'FAIL'}")

    print()
    print(f"FINAL RELEASE STATUS: {release_status.value.upper()}")

    if release_status == ReleaseStatus.REVIEW:
        print()
        print("Recommendation:")
        print("  Investigate the groundedness regression before release.")
        print(
            "  The candidate improves retrieval precision, context "
            "precision, latency, and cost while preserving recall, "
            "correctness, and relevance."
        )

    print("=" * 70)


def main() -> None:
    print("Loading documents...")
    chunks = load_documents()

    print(f"Loaded {len(chunks)} chunks.")

    print("Loading golden dataset...")
    cases = load_cases()

    print(f"Loaded {len(cases)} evaluation cases.")

    print("Building embedding service...")
    embedding_service = build_embedding_service()

    print("Embedding chunks...")
    embedded_chunks = embedding_service.embed_chunks(chunks)

    print("Building retrieval pipeline...")
    retrieval_pipeline = build_retrieval_pipeline(
        embedding_service=embedding_service,
        embedded_chunks=embedded_chunks,
    )

    print("Building shared LLM provider...")
    llm_provider = build_llm_provider()

    print("Building RAG pipeline...")
    rag_pipeline = build_rag_pipeline(
        retrieval_pipeline=retrieval_pipeline,
        llm_provider=llm_provider,
    )

    print("Building evaluation workflow...")
    workflow = build_evaluation_workflow(
        llm_provider=llm_provider,
    )

    # V1
    v1_report = run_evaluation(
        rag_pipeline=rag_pipeline,
        workflow=workflow,
        cases=cases,
        config=V1_CONFIG,
    )

    # V2
    v2_report = run_evaluation(
        rag_pipeline=rag_pipeline,
        workflow=workflow,
        cases=cases,
        config=V2_CONFIG,
    )

    # Reports
    print_report(
        name="V1",
        report=v1_report,
    )

    print_report(
        name="V2",
        report=v2_report,
    )

    print("Evaluate the reports against quality threshold ...")
    # V1
    v1_decision = evaluate_quality_gate(
        workflow=workflow,
        report=v1_report,
    )
    # V2
    v2_decision = evaluate_quality_gate(
        workflow=workflow,
        report=v2_report,
    )

    print("Compare the reports ...")
    # Compare the v1 vs v2 report and decision
    comparison = workflow.compare(
        v1_report=v1_report,
        v2_report=v2_report,
        v1_decision=v1_decision,
        v2_decision=v2_decision,
    )

    print_comparison(comparison)

    # Regresion Threshold for comparision
    regression_thresholds = build_regression_thresholds()

    print("Evaluating for regression ...")
    regression_decision = workflow.regression_gate(
        comparison=comparison,
        thresholds=regression_thresholds,
    )

    print_regression_decision(regression_decision)

    # Evluate for release decision
    release_status = determine_release_status(
        v2_decision=v2_decision,
        regression_decision=regression_decision,
    )

    print_release_decision(release_status)

    # Generate Release Report
    print_release_report(
        v1_report,
        v2_report,
        comparison,
        v1_decision,
        v2_decision,
        regression_decision,
        release_status,
    )


if __name__ == "__main__":
    main()
