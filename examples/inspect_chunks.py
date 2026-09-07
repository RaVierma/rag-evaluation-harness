from pathlib import Path

from rag_evaluation.chunking import TextChunker
from rag_evaluation.dataset import TextDocumentLoader

documents_path = Path("dataset/documents")

loader = TextDocumentLoader(documents_path)
documents = loader.load()

chunker = TextChunker(
    chunk_size=300,
    overlap=50,
)

for document in documents:
    chunks = chunker.chunk(document)

    print(f"\n{'=' * 80}")
    print(f"Document: {document.id}")
    print(f"Chunks:   {len(chunks)}")
    print(f"{'=' * 80}")

    for chunk in chunks:
        print(
            f"\n[{chunk.id}] "
            f"section={chunk.metadata.section!r} "
            f"length={len(chunk.content)}"
        )

        print(chunk.content)
