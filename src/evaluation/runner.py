from evaluation.diagnosis import diagnose
from evaluation.judges.base import Judge
from evaluation.metrics.context import context_precision, context_recall
from evaluation.metrics.retrieval import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.models.cases import (
    EvaluationCase,
)
from evaluation.models.reports import EvaluationCaseResult
from rag.models.outputs import SystemOutput
from rag.models.performance import RAGPerformance


class EvaluationRunner:
    def __init__(self, judge: Judge):
        self.judge = judge

    def evaluate(
        self,
        case: EvaluationCase,
        system_output: SystemOutput,
        rag_performance: RAGPerformance,
    ) -> EvaluationCaseResult:
        recall = recall_at_k(
            case.relevant_document_ids, system_output.retrieved_document_ids, k=20
        )

        precision = precision_at_k(
            case.relevant_document_ids, system_output.retrieved_document_ids, k=20
        )

        rr = reciprocal_rank(
            case.relevant_document_ids, system_output.retrieved_document_ids
        )

        ctx_recall = context_recall(
            case.relevant_document_ids, system_output.context_document_ids
        )

        ctx_precision = context_precision(
            case.relevant_document_ids, system_output.context_document_ids
        )

        generation = self.judge.evaluate(
            case.question,
            system_output.context,
            case.expected_answer,
            system_output.generated_answer,
        )

        diagnosis = diagnose(
            retrieval_recall=recall,
            groundedness=generation.groundedness.score,
            correctness=generation.correctness.score,
            relevance=generation.relevance.score,
        )

        return EvaluationCaseResult(
            case_id=case.id,
            system_output=system_output,
            retrieval_recall=recall,
            retrieval_precision=precision,
            retrieval_rr=rr,
            context_recall=ctx_recall,
            context_precision=ctx_precision,
            generation=generation,
            performance=rag_performance,
            diagnosis=diagnosis,
        )
