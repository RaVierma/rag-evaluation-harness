from rag.chunking import TextChunker
from rag.models.documents import Document, MetaData


def create_document(content: str) -> Document:
    return Document(
        id="doc-test",
        source="test.txt",
        content=content,
        metadata=MetaData(
            title="Test",
            department="engineering",
            version="1.0",
        ),
    )


def test_chunk_paragraph_smaller_than_chunk_size():
    document = create_document("# Introduction\n\nPython is required.")

    chunker = TextChunker(chunk_size=300, overlap=50)

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == "Python is required."
    assert chunks[0].document_id == "doc-test"
    assert chunks[0].position == 0
    assert chunks[0].metadata.section == "Introduction"


def test_chunk_multiple_sentences():
    document = create_document(
        "# Requirements\n\nPython is required. Rust is optional. Java is not required."
    )

    chunker = TextChunker(chunk_size=50, overlap=10)

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2

    assert all(len(chunk.content) <= 50 for chunk in chunks)


def test_chunk_oversized_sentence():
    sentence = "A" * 400

    document = create_document(f"# Requirements\n\n{sentence}")

    chunker = TextChunker(chunk_size=300, overlap=50)

    chunks = chunker.chunk(document)

    assert len(chunks) == 2

    assert len(chunks[0].content) == 300
    assert len(chunks[1].content) == 150

    assert all(len(chunk.content) <= 300 for chunk in chunks)


def test_chunk_ids_and_positions_are_deterministic():
    document = create_document(
        "# Requirements\n\nPython is required.\n\n# Installation\n\nInstall uv."
    )

    chunker = TextChunker(chunk_size=300, overlap=50)

    chunks = chunker.chunk(document)

    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))

    assert [chunk.id for chunk in chunks] == [
        f"doc-test-chunk-{i:04d}" for i in range(len(chunks))
    ]


def test_repeated_section_names_are_not_merged():
    document = create_document(
        "# Requirements\n\nPython is required.\n\n# Requirements\n\nRust is optional."
    )

    chunker = TextChunker(chunk_size=300, overlap=50)

    chunks = chunker.chunk(document)

    assert len(chunks) == 2

    assert chunks[0].metadata.section == "Requirements"
    assert chunks[1].metadata.section == "Requirements"

    assert chunks[0].content == "Python is required."
    assert chunks[1].content == "Rust is optional."
