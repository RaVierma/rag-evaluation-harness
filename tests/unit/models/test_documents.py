import pytest
from pydantic import ValidationError

from rag_evaluation.models import Document, MetaData


def test_document_creation():
    document = Document(
        id="doc-001",
        source="policy.txt",
        content="Employees may work remotely.",
        metadata=MetaData(
            title="Remote Work Policy",
            department="HR",
            version="2026",
        ),
    )

    assert document.id == "doc-001"
    assert document.source == "policy.txt"
    assert document.content == "Employees may work remotely."
    assert document.metadata.title == "Remote Work Policy"


def test_document_model_for_missing_id():
    with pytest.raises(ValidationError):
        document = Document(
            source="policy.txt",
            content="Employees may work remotely.",
            metadata=MetaData(
                title="Remote Work Policy",
                department="HR",
                version="2026",
            ),
        )

        Document(**document)


def test_document_model_for_missing_content():
    with pytest.raises(ValidationError):
        document = Document(
            id="doc-001",
            source="policy.txt",
            metadata=MetaData(
                title="Remote Work Policy",
                department="HR",
                version="2026",
            ),
        )

        Document(**document)


def test_document_model_for_missing_metadata():
    with pytest.raises(ValidationError):
        document = Document(
            id="doc-001",
            source="policy.txt",
            content="Employees may work remotely.",
        )

        Document(**document)


def test_document_model_for_department_missing_in_metadata():
    with pytest.raises(ValidationError):
        document = Document(
            id="doc-001",
            source="policy.txt",
            content="Employees may work remotely.",
            metadata=MetaData(
                title="Remote Work Policy",
                version="2026",
            ),
        )

        Document(**document)
