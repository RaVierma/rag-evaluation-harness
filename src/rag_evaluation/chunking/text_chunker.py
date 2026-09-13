import re

from .base import Chunker
from rag_evaluation.models import Chunk, Document, ChunkMetaData, Section

HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class TextChunker(Chunker):
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []

        # Section Detection
        sections = self.extract_sections(document.content)
        position = 0

        for section in sections:
            section_name = section.name

            # Paragraph Extraction
            paragraphs = self.extract_paragraphs(section.content)

            # Paragraph Packing: pack the small paragraph put together preserve semantic boundary
            pack_paragraphs = self.pack_units(paragraphs, join_with="\n\n")

            for paragraph in pack_paragraphs:
                if len(paragraph) <= self.chunk_size:
                    chunk_contents = [paragraph]
                else:
                    # Sentence Splitting
                    sentences = self.split_sentences(paragraph)

                    # Sentence Packing: pack the small sentence put together preserve semantic boundary
                    chunk_contents = self.pack_units(sentences)

                for content in chunk_contents:
                    if len(content) <= self.chunk_size:
                        contents = [content]
                    else:
                        # Hard Split (fallback)
                        contents = self.hard_split(content)

                    for chunk_content in contents:
                        chunks.append(
                            Chunk(
                                id=f"{document.id}-chunk-{position:04d}",
                                document_id=document.id,
                                content=chunk_content,
                                position=position,
                                metadata=ChunkMetaData(
                                    section=section_name.strip(), page=0
                                ),
                            )
                        )
                        position += 1

        return chunks

    def extract_sections(self, doc: str) -> list[Section]:
        sections: list[Section] = []

        current_section_name = "default"
        current_lines: list[str] = []

        for line in doc.splitlines(keepends=True):
            match = HEADING_PATTERN.match(line)

            if match:
                # Finish the current section.
                if current_lines:
                    sections.append(
                        Section(
                            name=current_section_name,
                            content="".join(current_lines).strip(),
                        )
                    )

                # Start a new section.
                current_section_name = match.group(1).strip()
                current_lines = []
                continue

            # Ignore leading blank lines.
            if not current_lines and not line.strip():
                continue

            current_lines.append(line)

        # Finish the final section.
        if current_lines:
            sections.append(
                Section(
                    name=current_section_name, content="".join(current_lines).strip()
                )
            )

        return sections

    def extract_paragraphs(self, content: str) -> list[str]:
        return [
            paragraph.strip()
            for paragraph in content.split("\n\n")
            if paragraph.strip()
        ]

    def split_sentences(self, paragraph: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in SENTENCE_PATTERN.split(paragraph)
            if sentence.strip()
        ]

    def pack_units(self, units: list[str], join_with: str = " ") -> list[str]:
        chunks: list[str] = []
        current_chunk: list[str] = []

        for unit in units:
            candidate = join_with.join(current_chunk + [unit])

            if len(candidate) <= self.chunk_size:
                current_chunk.append(unit)
            else:
                if current_chunk:
                    chunks.append(join_with.join(current_chunk))

                current_chunk = [unit]

        if current_chunk:
            chunks.append(join_with.join(current_chunk))

        return chunks

    def hard_split(self, sentence: str) -> list[str]:
        chunks: list[str] = []
        step = self.chunk_size - self.overlap
        start = 0

        while start < len(sentence):
            end = start + self.chunk_size
            content = sentence[start:end]

            start += step

            chunks.append(content)

        return chunks
