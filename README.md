# RAG Evaluation Harness

> **A production-oriented evaluation framework for measuring, comparing, and regression-testing Retrieval-Augmented Generation (RAG) systems across retrieval quality, context quality, answer quality, latency, and cost.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.x-e92063.svg)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Building a RAG system is not just a retrieval problem.

A system can retrieve more relevant documents while producing worse answers. It can improve answer quality while increasing latency and cost. It can pass simple questions while failing on multi-document, no-answer, permission-sensitive, or adversarial queries.

This project provides an evaluation layer for identifying those trade-offs **before changes are released to production**.

The core idea is:

```text
                    RAG System
                        │
                        ▼
                  System Output
                        │
                        ▼
               ┌─────────────────┐
               │ Evaluation       │
               │ Harness          │
               └────────┬────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Retrieval      Context      Generation
       Quality       Quality        Quality
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                 System Metrics
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
    Quality Gate                Regression Gate
          │                           │
          └─────────────┬─────────────┘
                        ▼
                 Ship / Investigate
```

The framework is intentionally separated from the RAG implementation itself. This allows the same evaluation harness to be used against different retrievers, rerankers, prompts, models, and RAG architectures.

---

# Why This Project?

A common approach to evaluating RAG systems is to ask:

> "Does the answer look good?"

That is not sufficient for production systems.

A useful evaluation process should answer:

* Did retrieval find the required evidence?
* Did reranking select the right context?
* Is the generated answer grounded in that context?
* Is the answer actually correct?
* Is the answer relevant to the user's question?
* Did latency increase?
* Did token usage increase?
* Did cost increase?
* Did a new version regress previously working scenarios?
* Should the new version be released?

This project turns those questions into measurable evaluation and release decisions.

---

# Key Capabilities

### Retrieval Evaluation

Measures retrieval performance using:

* Recall@K
* Precision@K
* Reciprocal Rank
* Mean Reciprocal Rank (MRR)

### Context Evaluation

Measures whether the final context supplied to the LLM contains useful evidence:

* Context Recall
* Context Precision

### Generation Evaluation

Evaluates the generated answer using:

* Groundedness
* Correctness
* Relevance

Generation evaluation supports an LLM-as-judge architecture with structured Pydantic validation.

### System Evaluation

Tracks operational characteristics:

* Average latency
* P95 latency
* Input tokens
* Output tokens
* Average cost

### Category-Level Evaluation

Evaluates performance by scenario type:

* Simple
* Difficult
* Multi-document
* Ambiguous
* No-answer
* Long-context
* Permission-sensitive
* Temporal
* Adversarial

### Quality Gates

Defines minimum quality and maximum operational thresholds.

Example:

```text
Retrieval Recall    >= 0.70
Groundedness        >= 0.80
Correctness         >= 0.80
P95 Latency         <= 1000 ms
Average Cost        <= $0.005
```

The system produces an explicit **pass/fail decision** rather than relying on manual inspection.

### Regression Gates

Compares two system versions and detects unacceptable regressions.

For example:

```text
v1 Correctness = 0.90
v2 Correctness = 0.87

Allowed degradation = 0.02

Actual delta = -0.03

Result = FAIL
```

The regression gate can independently protect quality metrics and operational metrics such as latency and cost.

---

# Architecture

```text
                         ┌──────────────────────┐
                         │   Evaluation Dataset │
                         │                      │
                         │  golden.jsonl        │
                         │  source documents    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    RAG System        │
                         │                      │
                         │ Retriever            │
                         │ Reranker             │
                         │ Context Builder      │
                         │ LLM                  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    SystemOutput      │
                         │                      │
                         │ Retrieved docs       │
                         │ Context docs         │
                         │ Context              │
                         │ Generated answer     │
                         │ Latency              │
                         │ Tokens               │
                         │ Cost                 │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Evaluation Runner    │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
      Retrieval Metrics      Context Metrics       Generation Judge
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Evaluation Report    │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
              Quality Decision              v1 / v2 Comparison
                                                   │
                                                   ▼
                                           Regression Decision
                                                   │
                                                   ▼
                                             Ship / Reject
```

---

# Core Design Principles

## 1. Evaluate the System, Not Just the LLM

RAG quality is a pipeline property.

```text
Question
   ↓
Retrieval
   ↓
Reranking
   ↓
Context Selection
   ↓
Prompt Construction
   ↓
LLM Generation
   ↓
Final Answer
```

A generation-only evaluation can hide retrieval failures.

This project therefore evaluates multiple stages independently.

---

## 2. Separate Retrieval From Generation

Retrieval metrics answer:

> "Did we retrieve the evidence?"

Generation metrics answer:

> "Did the model produce a good answer using the available evidence?"

Keeping these dimensions separate makes failures diagnosable.

For example:

```text
Retrieval Recall = 0.95
Groundedness     = 0.60
```

This suggests that the required evidence was retrieved but something later in the pipeline is causing problems.

---

## 3. Metrics Do Not Make the Decision

A higher metric does not automatically mean a better release.

A production decision must consider:

```text
Quality
  +
Latency
  +
Cost
  +
Reliability
  +
Security
```

For example:

```text
              v1          v2

Recall        0.80        0.90    ↑
Correctness   0.90        0.84    ↓
P95 latency   700 ms      1400 ms  ↑
Cost          $0.003      $0.008   ↑
```

Even though retrieval improved, the overall release may still be rejected.

---

## 4. Use the Same Dataset for Version Comparisons

Version comparisons should use identical evaluation cases:

```text
Golden Dataset
      │
      ├──────────► RAG v1
      │
      └──────────► RAG v2
```

This makes metric differences attributable to system changes rather than dataset changes.

---

## 5. Treat LLM Output as Untrusted Data

LLM judges produce structured output, but model output is not inherently trustworthy.

The framework therefore validates judge output through Pydantic models.

```text
LLM Judge
    │
    ▼
JSON
    │
    ▼
Pydantic Validation
    │
    ▼
GenerationEvaluation
```

Invalid JSON or invalid schemas are treated as evaluation failures rather than silently accepted.

---

# Evaluation Model

The framework separates evaluation into four major layers.

## Retrieval

```text
Recall@K
Precision@K
Reciprocal Rank
MRR
```

These measure whether relevant documents were retrieved and how highly they were ranked.

## Context

```text
Context Recall
Context Precision
```

These evaluate the documents that actually make it into the LLM context.

This distinction is important because:

```text
Retrieved Documents
        ↓
      Reranker
        ↓
Context Documents
        ↓
     LLM Context
```

A document can be successfully retrieved but still be excluded from the final context.

## Generation

```text
Groundedness
Correctness
Relevance
```

These evaluate the generated response.

### Groundedness

Are the claims supported by the supplied context?

### Correctness

Does the answer match the expected answer semantically?

### Relevance

Does the answer directly address the user's question?

---

# Evaluation Dataset

The project includes a small synthetic company-policy dataset designed to exercise different RAG failure modes.

```text
dataset/
├── README.md
├── golden.jsonl
├── documents/
│   ├── doc-001.txt
│   ├── ...
│   └── doc-011.txt
└── examples/
    └── sample_cases.jsonl
```

The dataset currently contains:

* 11 source documents
* 10 golden evaluation cases
* 4 sample cases
* 9 evaluation categories

The dataset intentionally includes:

* Single-document questions
* Multi-document questions
* Difficult questions
* No-answer queries
* Temporal queries
* Permission-sensitive queries
* Long-context scenarios
* Adversarial prompts

See [`dataset/README.md`](dataset/README.md) for the complete dataset design and schema.

---

# Project Structure

```text
rag-evaluation-harness/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── dataset/
│   ├── README.md
│   ├── golden.jsonl
│   ├── documents/
│   └── examples/
│       └── sample_cases.jsonl
│
├── examples/
│   ├── basic_evaluation.py
│   ├── compare_versions.py
│   └── regression_check.py
│
├── src/
│   └── rag_evaluation/
│       ├── __init__.py
│       │
│       ├── models/
│       │   ├── cases.py
│       │   ├── outputs.py
│       │   ├── generation.py
│       │   ├── reports.py
│       │   └── decisions.py
│       │
│       ├── metrics/
│       │   ├── retrieval.py
│       │   ├── context.py
│       │   └── generation.py
│       │
│       ├── judges/
│       │   ├── base.py
│       │   └── dummy.py
│       │
│       ├── providers/
│       │   ├── base.py
│       │   └── dummy.py
│       │
│       ├── evaluation/
│       │   ├── runner.py
│       │   ├── aggregator.py
│       │   ├── report.py
│       │   ├── category.py
│       │   ├── comparison.py
│       │   ├── quality_gate.py
│       │   └── regression_gate.py
│       │
│       ├── dataset/
│       │   └── loader.py
│       │
│       ├── prompts/
│       │   └── eval_prompt_v1.txt
│       │
│       └── utils/
│           └── helpers.py
│
└── tests/
    ├── unit/
    │   ├── metrics/
    │   ├── judges/
    │   ├── providers/
    │   └── evaluation/
    │
    └── fixtures/
        └── evaluation_cases.py
```

---

# Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

### Clone the repository

```bash
git clone https://github.com/<your-username>/rag-evaluation.git
cd rag-evaluation
```

### Install dependencies

```bash
uv sync
```

### Activate the environment

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

---

# Running the Examples

The examples are deterministic and do not require an external LLM provider.

## Basic Evaluation

Runs the evaluation pipeline against the sample dataset.

```bash
uv run python examples/basic_evaluation.py
```

Conceptually:

```text
Sample Dataset
      ↓
Mock SystemOutput
      ↓
EvaluationRunner
      ↓
EvaluationReport
```

---

## Version Comparison

Compares two simulated RAG system versions.

```bash
uv run python examples/compare_versions.py
```

The comparison reports:

* Metric values for v1
* Metric values for v2
* Absolute delta
* Percentage delta
* Quality-gate decisions

---

## Regression Check

Runs the regression gate between two system versions.

```bash
uv run python examples/regression_check.py
```

The regression gate answers:

> "Did the new version introduce an unacceptable regression?"

---

# Testing

Run the complete test suite:

```bash
uv run pytest
```

Run with verbose output:

```bash
uv run pytest -v
```

The tests cover:

* Retrieval metrics
* Context metrics
* Generation evaluation models
* Provider behavior
* Judge behavior
* Dataset loading
* Evaluation runner
* Report aggregation
* Category aggregation
* Quality decisions
* Version comparison
* Regression gates
* Validation and failure paths

The evaluation framework is designed so that core metrics and decision logic can be tested deterministically without depending on an external LLM API.

---

# Example: Evaluation Case

An evaluation case defines the expected behavior.

```python
EvaluationCase(
    id="case-001",
    question="How many days per week can an employee work remotely?",
    expected_answer="Employees may work remotely up to three days per week.",
    relevant_document_ids=["doc-001"],
    category=EvaluationCaseCategory.SIMPLE,
)
```

The RAG system produces:

```python
SystemOutput(
    retrieved_document_ids=["doc-001", "doc-007"],
    context_document_ids=["doc-001"],
    context="...",
    generated_answer="Employees can work remotely up to three days per week.",
    latency_ms=420,
    input_tokens=850,
    output_tokens=24,
    cost_usd=0.0012,
)
```

The evaluation harness then calculates retrieval, context, and generation metrics.

---

# Example: Release Decision

Suppose a new retriever produces:

```text
Metric                  v1       v2

Retrieval Recall        0.82     0.91
Context Precision       0.84     0.72
Groundedness            0.91     0.78
Correctness             0.90     0.81
P95 Latency             620ms    980ms
Average Cost            $0.003   $0.006
```

Retrieval improved significantly.

However:

```text
Context Precision  ↓
Groundedness       ↓
Correctness        ↓
Latency            ↑
Cost               ↑
```

The correct engineering response is not:

> "Recall improved, therefore v2 is better."

Instead:

```text
Observation
    ↓
Retrieval improved
    ↓
Context quality decreased
    ↓
Answer quality decreased
    ↓
Latency and cost increased
    ↓
Investigate reranking/context construction
    ↓
Do not release until regression is understood
```

This is the primary engineering principle behind the project.

---

# Design Decisions

## Why `Judge` Is an Abstract Interface

The framework does not assume that every evaluation must use an LLM.

```text
                 Judge
                   │
          ┌────────┴────────┐
          ▼                 ▼
      LLMJudge          RuleBasedJudge
          │
          ▼
      LLMProvider
```

This makes the evaluation layer independent from the judge implementation.

---

## Why `LLMProvider` Is Separate

The judge should not be tightly coupled to a specific model provider.

The abstraction allows implementations such as:

```text
LLMProvider
    ├── OpenAI
    ├── AWS Bedrock
    ├── Anthropic
    └── Test / Dummy Provider
```

This also makes testing deterministic.

---

## Why `SystemOutput` Is Separate From `EvaluationCase`

An evaluation case describes:

> What should happen?

A system output describes:

> What actually happened?

Keeping these separate allows the same evaluation cases to be reused across multiple system versions.

---

## Why the Evaluation Runner Does Not Execute RAG

The evaluation harness intentionally does not own the RAG pipeline.

Instead:

```text
RAG System
    ↓
SystemOutput
    ↓
Evaluation Harness
```

This keeps the evaluation framework independent from:

* Vector databases
* Embedding providers
* Retrievers
* Rerankers
* LLM providers
* Application frameworks

As a result, the harness can evaluate multiple RAG implementations using the same contract.

---

# Production-Oriented Considerations

This project is intentionally designed around concerns that appear in real RAG systems.

### Quality

```text
Retrieval → Context → Generation
```

### Reliability

Provider failures, invalid judge output, and validation errors are explicitly handled.

### Cost

Input tokens, output tokens, and estimated request cost are part of the evaluation report.

### Latency

Average and P95 latency are tracked because average latency alone can hide tail-performance problems.

### Regression Prevention

A system should not be considered improved merely because one metric increased.

Quality and operational thresholds can be applied before release.

### Observability

The output model preserves the measurements required to investigate why a system changed.

---

# Current Scope

The current implementation focuses on the **evaluation layer**.

It does not yet implement a production RAG pipeline.

Currently included:

* Evaluation dataset
* Retrieval metrics
* Context metrics
* Generation evaluation
* LLM judge abstraction
* LLM provider abstraction
* Evaluation runner
* Metric aggregation
* Category-level reporting
* Quality gates
* Version comparison
* Regression gates
* Deterministic examples
* Comprehensive unit tests

---

# Roadmap

The project is intentionally being built incrementally.

## Phase 1 — Evaluation Harness

* [x] Evaluation dataset
* [x] Retrieval metrics
* [x] Context metrics
* [x] Generation evaluation models
* [x] Judge abstraction
* [x] LLM provider abstraction
* [x] Evaluation runner
* [x] Report aggregation
* [x] Category-level evaluation
* [x] Quality gates
* [x] Version comparison
* [x] Regression gates
* [x] Unit tests
* [x] Deterministic examples

## Phase 2 — Real RAG Pipeline

* [ ] Document ingestion
* [ ] Document parsing
* [ ] Chunking
* [ ] Embeddings
* [ ] PostgreSQL + pgvector
* [ ] BM25 retrieval
* [ ] Hybrid retrieval
* [ ] Reranking
* [ ] Context construction
* [ ] LLM generation
* [ ] Real `SystemOutput` generation

## Phase 3 — Production Evaluation

* [ ] Real LLM judge
* [ ] Evaluation configuration
* [ ] Batch evaluation
* [ ] Experiment tracking
* [ ] Prompt/model versioning
* [ ] Evaluation result persistence
* [ ] CI regression checks
* [ ] Evaluation dashboards

## Phase 4 — Production RAG / LLMOps

* [ ] Tracing
* [ ] Token monitoring
* [ ] Cost monitoring
* [ ] Latency monitoring
* [ ] Error monitoring
* [ ] Prompt versioning
* [ ] Model fallback
* [ ] Retry policies
* [ ] Rate limiting
* [ ] Caching
* [ ] Continuous quality monitoring

---

# Engineering Approach

RAG improvements are treated as measurable experiments rather than
single-metric optimizations.

```text
Measure
   ↓
Understand
   ↓
Change
   ↓
Evaluate
   ↓
Compare
   ↓
Regression Check
   ↓
Release
```

The core investigation model is:

> **Observation → Hypothesis → Investigation → Root Cause → Fix → Trade-off → Decision**

A change is considered successful only when it improves the required
quality metrics without violating latency, cost, or regression constraints.

# Portfolio Value

This project demonstrates production-oriented RAG engineering beyond retrieval and prompt engineering.

It shows how to:

- Define measurable RAG quality criteria
- Build deterministic retrieval and context metrics
- Evaluate LLM-generated answers with structured judge outputs
- Compare RAG system versions using the same evaluation dataset
- Detect quality and operational regressions
- Apply release thresholds based on quality, latency, and cost
- Investigate failures across retrieval, reranking, context construction, and generation
- Separate evaluation infrastructure from the RAG implementation

The central engineering question is:

> **How do we know that a RAG system actually got better?**

The answer should come from reproducible evaluation and measurable trade-offs—not intuition.

---

# License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
