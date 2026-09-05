# Groundedness
#     ↓
# Are the claims supported by the provided context?
def groundedness(
    context: str,
    answer: str,
) -> float: ...


# Correctness
#     ↓
# Does the answer match the expected/ground-truth answer?
def correctness(
    expected_answer: str,
    generated_answer: str,
) -> float: ...


# Relevance
#     ↓
# Did the answer address the question?
def relevance(
    question: str,
    answer: str,
) -> float: ...
