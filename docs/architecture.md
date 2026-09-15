# Architecture

`rag-evaluation-harness` separates **RAG execution** from **RAG evaluation**.

```text
                 RAG Evaluation Harness
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
         `rag/`                 `evaluation/`
       RAG Runtime               Evaluation
            │                         │
     ┌──────┼──────┐          ┌───────┼────────┐
     │      │      │          │       │        │
 Ingestion Retrieval  LLM   Metrics  Judges   Gates
     │      │      │          │       │        │
     └──────┴──────┘          └───────┴────────┘
            │                         │
            ▼                         ▼
      RAGPipelineResult        EvaluationReport
                    \             /
                     ▼           ▼
                       Comparison
                           │
                           ▼
                    Release Decision
```

## Package Structure

```text
src/
├── common/
│   └── types.py
│
├── rag/
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── retriever/
│   ├── reranking/
│   ├── context/
│   ├── pipelines/
│   ├── providers/
│   ├── models/
│   └── prompts/
│
└── evaluation/
    ├── dataset/
    ├── metrics/
    ├── judges/
    ├── models/
    ├── prompts/
    ├── aggregator.py
    ├── comparison.py
    ├── quality_gate.py
    ├── regression_gate.py
    ├── release.py
    └── workflow.py
```

## Responsibilities

### `rag/`

Owns the RAG runtime:

* document ingestion
* chunking
* embeddings
* retrieval
* reranking
* context construction
* LLM generation
* pipeline execution

### `evaluation/`

Owns evaluation and release decisions:

* retrieval/context metrics
* generation evaluation
* LLM judges
* evaluation reports
* version comparison
* quality gates
* regression gates
* release decisions

### `common/`

Contains only genuinely shared primitives. It should remain intentionally small.

## Key Boundary

> **RAG executes. Evaluation measures and decides.**

The RAG layer should not depend on evaluation logic. Evaluation can consume RAG results to measure quality, performance, regressions, and release readiness.

## Design Principles

* Clear domain ownership
* Minimal shared code
* Measurable quality and performance
* Explicit quality/regression gates
* Evidence-based release decisions
* Avoid unnecessary abstraction and over-engineering
