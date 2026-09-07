from pathlib import Path

import pytest

from rag_evaluation.dataset import TextDocumentLoader


def test_text_document_loader():
    directory = Path("tests/unit/dataset/documents")

    loader = TextDocumentLoader(directory)

    documents = loader.load()

    assert len(documents) == 2


def test_text_document_loader_for_directory_not_exists():
    with pytest.raises(FileNotFoundError):
        directory = Path("tests/unit/dataset/document")

        loader = TextDocumentLoader(directory)

        loader.load()
