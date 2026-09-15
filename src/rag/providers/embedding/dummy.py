import hashlib
import math

from .base import EmbeddingProvider


class DummyEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "Dummy"

    @property
    def dimension(self) -> int:
        return 768

    def embed(self, text: str) -> tuple[float, ...]:
        if not text:
            raise ValueError("empty text not allowed.")

        values = self._generate_values(text)

        return self._normalize(values)

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:

        if not texts:
            return []

        if any(not text for text in texts):
            raise ValueError("empty text not allowed")

        return [self.embed(text) for text in texts]

    def stable_hash(self, text: str) -> bytes:
        return hashlib.sha256(text.encode("utf-8")).digest()

    def _byte_to_float(self, value: int) -> float:
        return (value / 255) * 2 - 1

    def _generate_values(
        self,
        text: str,
    ) -> tuple[float, ...]:
        embedding = []
        counters = math.ceil(self.dimension / 32)
        for counter in range(counters):
            if len(embedding) == self.dimension:
                break
            hsh = self.stable_hash(f"{text}:{counter}")
            for b in hsh:
                if len(embedding) == self.dimension:
                    break

                value = self._byte_to_float(b)

                embedding.append(value)

        return tuple(embedding)

    def _normalize(self, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            raise ValueError("values must be not empty")

        norm = math.sqrt(sum(value**2 for value in values))

        if norm == 0:
            raise ValueError("Cannot normalize a zero vector")

        return tuple(value / norm for value in values)
