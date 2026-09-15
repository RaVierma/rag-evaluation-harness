from pathlib import Path

from rag.ingestion.loader.base import DocumentLoader
from rag.models.documents import Document, MetaData


class TextDocumentLoader(DocumentLoader):
    def __init__(self, directory: Path):
        self.directory = directory

    def load(self) -> list[Document]:

        documents = []

        if not self.directory.exists():
            raise FileNotFoundError(f"{self.directory} not exists.")

        for file in self.directory.glob("*.txt"):
            document = Document(
                id=file.stem,
                source=file.name,
                content=file.read_text(encoding="utf-8"),
                metadata=MetaData(
                    title=file.stem,
                    department="unknown",
                    version="1.0",
                ),
            )

            documents.append(document)

        if not documents:
            raise ValueError(f"{self.directory} has no txt files.")

        return documents
