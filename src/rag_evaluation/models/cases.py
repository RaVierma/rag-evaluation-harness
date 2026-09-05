from enum import Enum

from pydantic import BaseModel

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


class EvaluationCase(BaseModel):
    id: NonEmptyString
    question: NonEmptyString
    expected_answer: NonEmptyString
    relevant_document_ids: list[str]
    category: EvaluationCaseCategory
