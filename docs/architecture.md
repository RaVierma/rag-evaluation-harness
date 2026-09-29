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
     ┌──────┼──────┐          ┌───────┼──────────┐
     │      │      │          │       │          │
 Ingestion Retrieval  LLM   Metrics  Judges   Reporting
     │      │      │          │       │          │
     └──────┴──────┘          └───────┴──────────┘
            │                         │
            ▼                         ▼
      RAG System Output       Evaluation Results
                                      │
                                      ▼
                              Evaluation Artifact
                                      │
                              ┌───────┴────────┐
                              │                │
                              ▼                ▼
                         Comparison          Gates
                              │                │
                              └───────┬────────┘
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
    ├── reporting/
    ├── artifact/
    ├── retrieval.py
    ├── relevance.py
    ├── diagnostics.py
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

* retrieval and ranking evaluation
* evaluation metrics
* context and generation evaluation
* LLM judges
* diagnostics
* evaluation reports
* evaluation artifacts
* version comparison
* quality gates
* regression gates
* release decisions

### `common/`

Contains only genuinely shared primitives. It should remain intentionally small.

## Key Boundary

> **RAG executes. Evaluation measures, interprets, and decides.**

The RAG layer should not depend on evaluation logic. Evaluation consumes RAG results to measure quality and performance, diagnose results, compare versions, preserve evidence, and determine release decisions.

## Design Principles

* Clear domain ownership
* Minimal shared code
* Measurable quality and performance
* Explicit quality and regression gates
* Evidence-based release decisions
* Preserve evaluation results as structured artifacts
* Avoid unnecessary abstraction and over-engineering
