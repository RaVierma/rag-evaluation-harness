# Design Decisions

## Decision 1 — Evaluation is separate from RAG execution
The evaluation harness consumes `SystemOutput` rather than executing the RAG pipeline itself.

**Reason:**  This allows the same evaluation framework to evaluate:
```
RAG v1
RAG v2
Different embedding models
Different rerankers
Different LLMs
Different retrieval strategies
```
without coupling evaluation logic to implementation details.

## Decision 2 — Typed LLM judge output
```
LLM
 ↓
JSON
 ↓
Pydantic validation
 ↓
GenerationEvaluation
```
**Reason:** LLM output is probabilistic and untrusted. The application boundary should be deterministic and typed.

## Decision 3 — Quality gate vs regression gate
Keep these separate.

```
Quality Gate
    ↓
"Is this version good enough?"

Regression Gate
    ↓
"Did this version degrade too much compared with baseline?"
```
