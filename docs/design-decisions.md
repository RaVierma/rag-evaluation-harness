# Design Decisions

## Decision 1 — Evaluation is separate from RAG execution

The evaluation framework is separated from the RAG runtime.

```text
RAG Runtime

    ↓

RAG Outputs / Results

    ↓

Evaluation
```

**Reason:** This allows the same evaluation framework to evaluate:

```text
RAG v1

RAG v2

Different embedding models

Different rerankers

Different LLMs

Different retrieval strategies
```

without coupling evaluation logic to RAG implementation details.

---

## Decision 2 — RAG owns runtime performance

`RAGPerformance` remains part of the RAG domain.

```text
RAG Pipeline

     ↓

RAGPerformance

     ↓

Evaluation
```

**Reason:** The RAG runtime produces execution measurements such as latency and token usage. Evaluation consumes those measurements to calculate metrics, compare versions, and make decisions.

---

## Decision 3 — Typed LLM judge output

LLM judge responses are converted into typed evaluation models.

```text
LLM
 ↓
JSON
 ↓
Pydantic validation
 ↓
GenerationEvaluation
```

**Reason:** LLM output is probabilistic and untrusted. The application boundary should be deterministic and typed.

---

## Decision 4 — Quality gate vs regression gate

Keep these decisions separate.

```text
Quality Gate

    ↓

"Is this version good enough?"

Regression Gate

    ↓

"Did this version degrade compared with the baseline?"
```

**Reason:** Absolute quality and change-from-baseline are different questions and should have independent policies.

---

## Decision 5 — Keep shared code minimal

Only genuinely cross-domain primitives belong in `common/`.

```text
common/

└── types.py
```

**Reason:** Prevent `common/` from becoming a dumping ground for unrelated utilities and constants. Domain-specific functionality should remain inside `rag/` or `evaluation/`.

---

## Decision 6 — Explicit domain ownership

The repository uses explicit ownership:

```text
rag/

    RAG execution

evaluation/

    Measurement, diagnosis, reporting, and release decisions

common/

    Shared primitives only
```

**Reason:** Clear ownership makes the codebase easier to understand, test, extend, and evolve without unnecessary abstractions.

---

## Decision 7 — Preserve evaluation evidence as an Artifact

Evaluation results are preserved as a structured Artifact rather than treated as temporary output.

```text
Evaluation Run

     ↓

Evaluation Artifact

     ↓

Comparison / Gates
```

**Reason:** A stable Artifact allows evaluation results to be inspected, compared, reproduced, and consumed by downstream quality and regression decisions.
