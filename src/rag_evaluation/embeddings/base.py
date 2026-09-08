from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, text: str) -> tuple[float, ...]: ...

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
    ) -> list[tuple[float, ...]]: ...
