from rag_evaluation.judges.llm import LLMJudge
from rag_evaluation.providers.generation.ollama_com import OllamaLLMProvider

test_cases = [
    {
        "name": "perfect",
        "question": "How many vacation days can employees carry forward?",
        "context": "Employees may carry forward up to five unused vacation days\ninto the next calendar year.",
        "expected_answer": "Employees may carry forward up to five unused vacation days.",
        "generated_answer": "Employees may carry forward up to five unused vacation days.",
    },
    {
        "name": "Hallucination",
        "question": "How many vacation days can employees carry forward?",
        "context": "Employees may carry forward up to five unused vacation days.",
        "expected_answer": "Employees may carry forward up to five unused vacation days.",
        "generated_answer": "Employees may carry forward up to ten unused vacation days.",
    },
    {
        "name": "Correct but ungrounded",
        "question": "How many vacation days can employees carry forward?",
        "context": "Employees must obtain manager approval for remote work.",
        "expected_answer": "Employees may carry forward up to five vacation days.",
        "generated_answer": "Employees may carry forward up to five vacation days.",
    },
    {
        "name": "Irrelevant answer",
        "question": "How many vacation days can employees carry forward?",
        "context": "Employees may carry forward up to five unused vacation days.",
        "expected_answer": "Employees may carry forward up to five unused vacation days.",
        "generated_answer": "The annual leave policy is version 2026.1.",
    },
]

provider = OllamaLLMProvider(model_name="mistral:7b-instruct-v0.3-q4_K_M")

# res = provider.call("""You are a strict evaluation judge.

# Evaluate ONLY the relevance of the Generated Answer to the Question.

# Question:
# How many vacation days can be carried forward?

# Generated Answer:
# The annual leave policy is version 2026.1.

# Rules:

# - Relevance = 1.0 ONLY if the Generated Answer directly provides
#   the information requested by the Question.
# - Relevance = 0.0 if the Generated Answer does not provide the
#   information requested by the Question.
# - Do NOT consider whether the answer is related to the same topic.
# - Do NOT infer missing information.
# - The question asks for a NUMBER.
# - The generated answer provides NO NUMBER answering the question.
# - Therefore the correct score is 0.0.

# Return only:

# {
#   "score": 0.0,
#   "reason": "The answer does not provide the requested number of vacation days."
# }""")
# print(res.content)

judge = LLMJudge(provider)
for test_case in test_cases:
    if test_case["name"] != "Irrelevant answer":
        continue
    for i in range(5):
        evaluation = judge.evaluate(
            question=test_case["question"],
            context=test_case["context"],
            expected_answer=test_case["expected_answer"],
            generated_answer=test_case["generated_answer"],
        )

        print(test_case["name"])
        print(evaluation.model_dump_json(indent=2))
