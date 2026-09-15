from rag.models.chunks import HybridRetrievedChunk, RerankedChunk
from rag.reranking.base import Reranker


class DummyReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[HybridRetrievedChunk],
        top_k: int,
    ) -> list[RerankedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if not candidates:
            return []

        if top_k <= 0:
            raise ValueError("top k must be greater than zero")

        rerank_chunks: list[RerankedChunk] = []

        for candidate in candidates:
            score = self._score(query, candidate.chunk.content)

            rerank_chunks.append(RerankedChunk(chunk=candidate, rerank_score=score))

        return sorted(rerank_chunks, key=lambda x: x.rerank_score, reverse=True)[:top_k]

    def _score(self, query: str, content: str) -> float:
        if not query.strip():
            raise ValueError("query must not be empty")

        if not content.strip():
            raise ValueError("content must not be empty")

        query_terms = self._tokenize(query)
        content_terms = self._tokenize(content)

        overlap = query_terms & content_terms

        score = len(overlap) / len(query_terms)

        return score

    def _tokenize(self, text: str) -> set[str]:
        if not text.strip():
            return set()

        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "are",
            "can",
            "to",
            "of",
            "is",
            "who",
            "what",
            "how",
            "for",
            "in",
            "on",
        }

        text = text.lower()

        for char in ".,?!:;()[]{}":
            text = text.replace(char, " ")

        return {token for token in text.split() if token not in stop_words}
