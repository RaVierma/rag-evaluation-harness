from pydantic import BaseModel

from rag_evaluation.types.strings import NonEmptyString


class MetaData(BaseModel):
    title: NonEmptyString
    department: NonEmptyString
    version: NonEmptyString


class Document(BaseModel):
    id: NonEmptyString
    source: NonEmptyString
    content: NonEmptyString
    metadata: MetaData


class Section(BaseModel):
    name: NonEmptyString
    content: NonEmptyString
