# Evaluation Methodology

The evaluation process uses a fixed dataset and protocol to measure RAG quality and system performance consistently.

```text
Golden Dataset
      ↓
Fixed Evaluation Protocol
      ↓
Run RAG System
      ↓
Collect Results
      ↓
Measure
      ↓
Compare with Baseline
      ↓
Quality / Regression Gates
      ↓
Release Decision
```

## Evaluation Dimensions

The harness evaluates four main areas:

```text
Retrieval
  → Recall@K, Precision@K, MRR

Context
  → Context Recall, Context Precision

Generation
  → Groundedness, Correctness, Relevance

Performance
  → Latency, P95, Tokens, Cost
```

## Evaluation Principle

> **Measure first, compare changes, then make a release decision.**

The same evaluation protocol should be used across RAG versions so that improvements and regressions are measurable and reproducible.
