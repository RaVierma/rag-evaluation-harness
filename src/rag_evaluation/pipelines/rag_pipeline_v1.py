import time

from rag_evaluation.context.builder import ContextBuilder
from rag_evaluation.models import RAGPerformance, RAGPipelineResult, SystemOutput
from rag_evaluation.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_evaluation.providers.generation.base import LLMProvider
from rag_evaluation.utils.helper import load_generation_prompt


class RAGPipeline:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider,
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.context_builder = context_builder
        self.llm_provider = llm_provider

    def run(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 5,
    ) -> RAGPipelineResult:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        total_start = time.monotonic()

        retrieval_start = time.monotonic()
        retrieval_output = self.retrieval_pipeline.retrieve(
            query, candidate_k=candidate_k, top_k=top_k
        )
        retrieval_latency_ms = (time.monotonic() - retrieval_start) * 1000

        context_start = time.monotonic()
        context = self.context_builder.build(retrieval_output.results)
        context_latency_ms = (time.monotonic() - context_start) * 1000

        prompt = load_generation_prompt("v1", query, context)

        llm_start = time.monotonic()
        llm_output = self.llm_provider.call(prompt)
        llm_latency_ms = (time.monotonic() - llm_start) * 1000

        if not llm_output.success or not llm_output.content:
            raise RuntimeError(llm_output.error or "LLM call failed")

        total_latency_ms = (time.monotonic() - total_start) * 1000

        document_ids = [
            chunk.chunk.chunk.document_id for chunk in retrieval_output.results
        ]

        output = SystemOutput(
            retrieved_document_ids=document_ids,
            context_document_ids=document_ids,
            context=context,
            generated_answer=llm_output.content,
        )

        generation_tokens_per_second = (
            llm_output.output_tokens / (llm_latency_ms / 1000)
            if llm_output.output_tokens is not None and llm_latency_ms > 0
            else None
        )

        performance = RAGPerformance(
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            context_latency_ms=context_latency_ms,
            llm_latency_ms=llm_latency_ms,
            input_tokens=llm_output.input_tokens,
            output_tokens=llm_output.output_tokens,
            generation_tokens_per_second=generation_tokens_per_second,
            cost_usd=llm_output.cost_usd,
        )

        return RAGPipelineResult(output=output, performance=performance)
