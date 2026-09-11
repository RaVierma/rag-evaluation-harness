from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from rag_evaluation.types.strings import NonEmptyString


class EvaluationCaseCategory(str, Enum):
    SIMPLE = "simple"
    DIFFICULT = "difficult"
    MULTI_DOCUMENT = "multi_document"
    AMBIGUOUS = "ambiguous"
    NO_ANSWER = "no_answer"
    LONG_CONTEXT = "long_context"
    PERMISSION_SENSITIVE = "permission_sensitive"
    TEMPORAL = "temporal"
    ADVERSARIAL = "adversarial"


class RelevanceJudgment(BaseModel):
    chunk_id: NonEmptyString
    relevance: Annotated[int, Field(ge=0, le=3, strict=True)]


class EvaluationCase(BaseModel):
    id: NonEmptyString
    question: NonEmptyString
    expected_answer: NonEmptyString
    relevant_document_ids: list[NonEmptyString]
    relevance_judgments: list[RelevanceJudgment]
    category: EvaluationCaseCategory

    @field_validator("relevance_judgments", mode="after")
    @classmethod
    def validate_relevance_judgments(
        cls,
        relevance_judgments: list[RelevanceJudgment],
    ) -> list[RelevanceJudgment]:
        chunk_ids = [judgment.chunk_id for judgment in relevance_judgments]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("duplicate relevance judgment found")

        return relevance_judgments
