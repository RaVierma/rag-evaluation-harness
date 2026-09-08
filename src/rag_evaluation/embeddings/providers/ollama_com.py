import os

from ollama import Client

from rag_evaluation.embeddings import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, dimension: int):
        self._model_name = model_name
        self._dimension = dimension
        self._client = Client(
            # host="nomic-embed-text:latest",
            # headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> tuple[float, ...]:
        if not text:
            raise ValueError("empty text not allowed.")

        response = self._client.embed(
            model=self.model_name, input=text, dimensions=self.dimension
        )

        vector = response["embeddings"][0]

        if len(vector) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.dimension}, got {len(vector)}"
            )

        return tuple(vector)

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[tuple[float, ...]]:
        if not texts:
            return []

        if any(not text for text in texts):
            raise ValueError("empty text not allowed")

        response = self._client.embed(
            model=self.model_name, input=texts, dimensions=self.dimension
        )

        vectors = response["embeddings"]

        if len(vectors) != len(texts):
            raise ValueError("Input ouput mismatch.")

        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: "
                    f"expected {self.dimension}, got {len(vector)}"
                )

        return [tuple(vector) for vector in vectors]
