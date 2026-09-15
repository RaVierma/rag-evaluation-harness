import pytest
from pydantic import ValidationError

from rag.models.chunks import Chunk, ChunkMetaData


def test_chunk_model_for_valid_data():

    chunk_data = Chunk(
        id="chk-0001",
        document_id="doc-0001",
        content="some test content",
        position=0,
        metadata=ChunkMetaData(section="Introduction", page=1),
    )

    assert chunk_data.id == "chk-0001"
    assert chunk_data.document_id == "doc-0001"
    assert chunk_data.content == "some test content"
    assert chunk_data.position == 0
    assert chunk_data.metadata.page == 1


def test_chunk_model_for_missing_chunk_id():

    with pytest.raises(ValidationError):
        Chunk(
            document_id="doc-0001",
            content="some test content",
            position=0,
            metadata=ChunkMetaData(section="Introduction", page=1),
        )


def test_chunk_model_for_missing_document_id():

    with pytest.raises(ValidationError):
        Chunk(
            id="chk-0001",
            content="some test content",
            position=0,
            metadata=ChunkMetaData(section="Introduction", page=1),
        )


def test_chunk_model_for_missing_content():

    with pytest.raises(ValidationError):
        Chunk(
            id="chk-0001",
            document_id="doc-0001",
            position=0,
            metadata=ChunkMetaData(section="Introduction", page=1),
        )


def test_chunk_model_for_missing_position():

    with pytest.raises(ValidationError):
        Chunk(
            id="chk-0001",
            document_id="doc-0001",
            content="some test content",
            metadata=ChunkMetaData(section="Introduction", page=1),
        )


def test_chunk_model_for_missing_metadata():

    with pytest.raises(ValidationError):
        Chunk(
            id="chk-0001",
            document_id="doc-0001",
            content="some test content",
            position=0,
        )
