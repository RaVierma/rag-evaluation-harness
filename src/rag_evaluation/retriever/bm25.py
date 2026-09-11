import math

from rag_evaluation.models import RetrievedChunk
from rag_evaluation.models.chunks import EmbeddedChunk


class BM25Retriever:
    def __init__(
        self,
        documents: list[EmbeddedChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._documents = documents
        self._k1 = k1
        self._b = b

        # Precompute corpus-level information
        self._doc_contents = [doc.content for doc in self._documents]
        self._tokenized_documents = {
            document.chunk_id: self._tokenize(document.content)
            for document in documents
        }
        self._document_frequencies = self._document_frequency(
            [content for content in self._doc_contents]
        )
        self._average_doc_length = self._average_document_length(self._doc_contents)

    def retrieve(
        self,
        query: str,
        candidate_k: int,
    ) -> list[RetrievedChunk]:

        if not query.strip():
            raise ValueError("query must not be empty")

        if candidate_k <= 0:
            raise ValueError("tok k must be greater that zero.")

        retrieve_chunks: list[RetrievedChunk] = []

        query_tokens = self._tokenize(query)
        document_count = len(self._documents)

        for embed_chunk in self._documents:
            document_tokens = self._tokenized_documents[embed_chunk.chunk_id]

            score = self.bm25_document_score(
                query_tokens,
                document_tokens,
                self._document_frequencies,
                document_count,
                self._average_doc_length,
            )
            retrieve_chunks.append(RetrievedChunk(chunk=embed_chunk, score=score))

        return sorted(retrieve_chunks, key=lambda x: x.score, reverse=True)[
            :candidate_k
        ]

    def _tokenize(self, text: str) -> list[str]:
        if not text.strip():
            return []

        text = text.lower()

        for char in ".,?!":
            text = text.replace(char, "")

        return text.split()

    def _document_frequency(
        self,
        documents: list[str],
    ) -> dict[str, int]:
        df: dict[str, int] = {}

        for document in documents:
            tokens = self._tokenize(document)

            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        return df

    def _inverse_document_frequency(
        self,
        document_count: int,
        document_frequency: int,
    ) -> float:

        if document_count <= 0:
            raise ValueError("document count not be zero")

        if document_frequency < 0:
            raise ValueError("document_frequency count not be zero")

        if document_frequency > document_count:
            raise ValueError("document_frequency must be less than number of document")

        return math.log(
            1
            + (
                (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            ),
        )

    def _term_frequency(
        self,
        tokens: list[str],
        term: str,
    ) -> int:

        if not len(tokens):
            raise ValueError("tokens must not empty")

        if not term:
            raise ValueError("term must not empty string.")

        return tokens.count(term)

    def _document_length(self, tokens: list[str]) -> int:
        return len(tokens)

    def _average_document_length(
        self,
        documents: list[str],
    ) -> float:
        if not documents:
            return 0.0

        return sum(len(self._tokenize(document)) for document in documents) / len(
            documents
        )

    def _bm25_term_score(
        self,
        term_frequency: int,
        document_frequency: int,
        document_count: int,
        document_length: int,
        average_document_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:

        if term_frequency < 0:
            raise ValueError("term_frequency must not be negative")

        if document_frequency < 0:
            raise ValueError("document_frequency must not be negative")

        if document_frequency > document_count:
            raise ValueError("document_frequency must not exceed document_count")

        if document_count <= 0:
            raise ValueError("document_count must be greater than zero")

        if document_length <= 0:
            raise ValueError("document_length must be greater than zero")

        if average_document_length <= 0:
            raise ValueError("average_document_length must be greater than zero")

        if k1 < 0:
            raise ValueError("k1 must not be negative")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        if term_frequency == 0:
            return 0.0

        idf = self._inverse_document_frequency(
            document_count,
            document_frequency,
        )

        length_normalization = 1 - b + b * (document_length / average_document_length)

        numerator = term_frequency * (k1 + 1)

        denominator = term_frequency + k1 * length_normalization

        return idf * (numerator / denominator)

    def bm25_document_score(
        self,
        query_tokens: list[str],
        document_tokens: list[str],
        document_frequencies: dict[str, int],
        document_count: int,
        average_document_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        if not query_tokens or not document_tokens:
            return 0.0

        score = 0.0
        for term in set(query_tokens):
            term_frequency: int = self._term_frequency(document_tokens, term)
            document_frequency: int = document_frequencies.get(term, 0)
            score += self._bm25_term_score(
                term_frequency,
                document_frequency,
                document_count,
                len(document_tokens),
                average_document_length,
                k1=k1,
                b=b,
            )

        return score
