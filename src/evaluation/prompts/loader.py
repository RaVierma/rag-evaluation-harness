def load_eval_prompt(
    version: str,
    question: str,
    context: str,
    expected_answer: str,
    generated_answer: str,
) -> str:
    with open(f"src/evaluation/prompts/judge/{version}.txt", "r") as f:
        return (
            f.read()
            .replace("{##question##}", question)
            .replace("{##context##}", context)
            .replace("{##expected_answer##}", expected_answer)
            .replace("{##generated_answer##}", generated_answer)
        )
