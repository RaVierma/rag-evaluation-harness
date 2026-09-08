from typing import Annotated

from pydantic import BaseModel, Field

from rag_evaluation.types.strings import NonEmptyString


class ChunkMetaData(BaseModel):
    section: NonEmptyString
    page: Annotated[int, Field(ge=0, strict=True)]


class Chunk(BaseModel):
    id: NonEmptyString
    document_id: NonEmptyString
    content: NonEmptyString
    position: Annotated[int, Field(ge=0, strict=True)]
    metadata: ChunkMetaData


class EmbeddedChunk(BaseModel):
    chunk_id: NonEmptyString
    document_id: NonEmptyString
    content: NonEmptyString
    embedding: tuple[float, ...]


class RetrievedChunk(BaseModel):
    chunk: EmbeddedChunk
    score: float
