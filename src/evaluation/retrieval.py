from evaluation.metrics.retrieval import recall_at_k, reciprocal_rank
from evaluation.models.cases import EvaluationCase
from evaluation.relevance import evaluate_ranking
from rag.models.pipeline import RetrievalOutput


def evaluate_retrieval(
    retrieval_output: RetrievalOutput,
    case: EvaluationCase,
    k: int,
) -> dict[str, float]:
    relevant_document_ids = case.relevant_document_ids
    retrieved_document_ids = [
        rerchk.chunk.chunk.chunk_id.split("-chunk-")[0]
        for rerchk in retrieval_output.results
    ]

    recall = recall_at_k(
        relevant_document_ids=relevant_document_ids,
        retrieved_document_ids=retrieved_document_ids,
        k=k,
    )

    rr = reciprocal_rank(
        relevant_document_ids=relevant_document_ids,
        retrieved_document_ids=retrieved_document_ids,
    )

    ndcg = evaluate_ranking(
        retrieved_chunks=retrieval_output.results,
        relevance_judgments=case.relevance_judgments,
        k=k,
    )

    return {"recall_at_k": recall, "rr_at_k": rr, "ndcg_at_k": ndcg}
