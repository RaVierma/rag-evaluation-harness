from pathlib import Path

from rag_evaluation.dataset import TextDocumentLoader


def main():
    document_directory = Path("dataset/documents")
    loader = TextDocumentLoader(document_directory)

    documents = loader.load()

    count = len(documents)
    print(f"\n Documents Loaded: {count}\n\n")

    print("ID", "Source", "\tTitle", "\tCharacters", sep="\t")
    print("-" * 50)
    for document in documents:
        print(
            f"{document.id} | {document.source} | {document.metadata.title} |\t{len(document.content)} chars"
        )


if __name__ == "__main__":
    main()
