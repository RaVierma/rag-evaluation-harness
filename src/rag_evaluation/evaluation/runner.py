from rag_evaluation.judges.base import Judge
from rag_evaluation.metrics.context import context_precision, context_recall
from rag_evaluation.metrics.retrieval import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_evaluation.models import EvaluationCase, EvaluationCaseResult, SystemOutput


class EvaluationRunner:
    def __init__(self, judge: Judge):
        self.judge = judge

    def evaluate(
        self, case: EvaluationCase, system_output: SystemOutput
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

        return EvaluationCaseResult(
            case_id=case.id,
            system_output=system_output,
            retrieval_recall=recall,
            retrieval_precision=precision,
            retrieval_rr=rr,
            context_recall=ctx_recall,
            context_precision=ctx_precision,
            generation=generation,
        )
