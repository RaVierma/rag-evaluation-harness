import time

from rag_evaluation.context.builder import ContextBuilder
from rag_evaluation.models.outputs import SystemOutput
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
    ) -> SystemOutput:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        start = time.monotonic()

        retrieval_output = self.retrieval_pipeline.retrieve(
            query, candidate_k=candidate_k, top_k=top_k
        )

        chunks = retrieval_output.results

        context = self.context_builder.build(chunks)
        prompt = load_generation_prompt("v1", query, context)

        llm_output = self.llm_provider.call(prompt)

        if not llm_output.success or not llm_output.content:
            raise RuntimeError(llm_output.error or "LLM call failed")

        end = time.monotonic()
        document_ids = [chunk.chunk.chunk.document_id for chunk in chunks]

        return SystemOutput(
            retrieved_document_ids=document_ids,
            context_document_ids=document_ids,
            context=context,
            generated_answer=llm_output.content,
            latency_ms=(end - start) * 1000,
            input_tokens=llm_output.input_tokens,
            output_tokens=llm_output.output_tokens,
            cost_usd=llm_output.cost_usd,
        )
