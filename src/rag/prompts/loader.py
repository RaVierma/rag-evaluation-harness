def load_generation_prompt(version: str, question: str, context: str) -> str:
    with open(f"src/rag/prompts/generation/{version}.txt", "r") as f:
        return (
            f.read()
            .replace("{##question##}", question)
            .replace("{##context##}", context)
        )
