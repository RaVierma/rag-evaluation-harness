from rag_evaluation.models import RerankedChunk


class ContextBuilder:
    def build(self, chunks: list[RerankedChunk]) -> str:
        if not chunks:
            return ""

        return "\n\n".join(
            f"[Chunk: {chunk.chunk.chunk.chunk_id}]\n{chunk.chunk.chunk.content}"
            for chunk in chunks
        )
