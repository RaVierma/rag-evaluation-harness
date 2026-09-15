from evaluation.metrics.ndcg import ndcg_at_k
from evaluation.models.cases import RelevanceJudgment
from rag.models.chunks import RerankedChunk, RetrievedChunk


def build_relevance_map(
    relevance_judgments: list[RelevanceJudgment],
) -> dict[str, int]:

    return {
        relevance_judgment.chunk_id: relevance_judgment.relevance
        for relevance_judgment in relevance_judgments
    }


def get_relevance_scores(
    retrieved_chunks: list[RetrievedChunk | RerankedChunk],
    relevance_map: dict[str, int],
) -> list[int]:
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks must not be empty")

    return [
        relevance_map.get(_get_chunk_id(retrieved_chunk), 0)
        for retrieved_chunk in retrieved_chunks
    ]


def evaluate_ranking(
    retrieved_chunks: list[RetrievedChunk | RerankedChunk],
    relevance_judgments: list[RelevanceJudgment],
    k: int,
) -> float:
    relevance_map = build_relevance_map(relevance_judgments)
    relevance_scores = get_relevance_scores(retrieved_chunks, relevance_map)

    return ndcg_at_k(relevance_scores, k)


def _get_chunk_id(
    retrieved_chunk: RetrievedChunk | RerankedChunk,
) -> str:
    if isinstance(retrieved_chunk, RetrievedChunk):
        return retrieved_chunk.chunk.chunk_id

    return retrieved_chunk.chunk.chunk.chunk_id
