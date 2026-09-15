# RAG Evaluation Harness

> **A production-oriented evaluation framework for measuring, comparing, and regression-testing RAG systems across retrieval quality, context quality, answer quality, latency, and cost.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/) [![Pydantic](https://img.shields.io/badge/Pydantic-2.x-e92063.svg)](https://docs.pydantic.dev/) [![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

RAG optimization is not a single-metric problem.

A change can improve retrieval while hurting groundedness, or reduce latency and cost while introducing a quality regression.

This project provides an evaluation workflow that makes those trade-offs measurable.

```text
RAG System
    ↓
System Output
    ↓
Evaluation
    ↓
Metrics
    ↓
Comparison
    ↓
Quality Gate + Regression Gate
    ↓
Release Decision
```

The core principle is:

> **Measure before optimizing.**

---

## Architecture

The project separates **RAG execution** from **evaluation**.

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

### RAG

Owns runtime execution:

* document ingestion
* chunking
* embeddings
* retrieval
* reranking
* context construction
* LLM generation
* pipeline execution

### Evaluation

Owns measurement and release decisions:

* retrieval and context metrics
* generation evaluation
* LLM judges
* reporting
* version comparison
* quality gates
* regression gates
* release decisions

> **RAG executes. Evaluation measures and decides.**

See [Architecture](docs/architecture.md).

---

## Evaluation Metrics

The harness evaluates four dimensions:

```text
Retrieval
├── Recall@K
├── Precision@K
├── Reciprocal Rank
└── MRR

Ranking
└── NDCG

Context
├── Context Recall
└── Context Precision

Generation
├── Groundedness
├── Correctness
└── Relevance

Performance
├── P50 / P95 Latency
├── Input Tokens
├── Output Tokens
└── Cost
```

See [Metrics](docs/metrics.md).

---

## Evaluation Workflow

A fixed golden dataset is used to compare system versions consistently.

```text
Golden Dataset
      ↓
Evaluation Protocol
      ↓
Run RAG System
      ↓
Collect Results
      ↓
Measure
      ↓
Compare with Baseline
      ↓
Quality Gate
      ↓
Regression Gate
      ↓
Release Decision
```

The same evaluation cases are used across versions so that metric changes can be attributed to system changes rather than dataset changes.

See [Evaluation Methodology](docs/evaluation-methodology.md).

---

## Retrieval Baseline

The initial retrieval experiment compared deterministic dummy embeddings with semantic embeddings.

| Metric   | Dummy | Semantic |
| -------- | ----: | -------: |
| Recall@3 |  0.00 |     0.80 |
| Recall@5 |  0.20 |     1.00 |

This established semantic embeddings as the retrieval baseline.

The retrieval evaluation was subsequently extended to:

```text
Dense Retrieval
       +
BM25
       ↓
RRF Fusion
       ↓
CrossEncoder Reranking
```

The expanded evaluation achieved:

| Metric   | Result |
| -------- | -----: |
| Recall@5 |  1.000 |
| RR@5     |  1.000 |
| NDCG@5   |  0.938 |

See [Retrieval Baseline](docs/retrieval-baseline.md).

---

## Final Evaluation: V1 vs V2

The final evaluation compares two system versions across quality, performance, and cost.

| Metric            |      V1 |      V2 |    Change |
| ----------------- | ------: | ------: | --------: |
| Recall@K          |    1.00 |    1.00 |         — |
| Precision@K       |    0.58 |    0.77 |     +0.19 |
| MRR               |    1.00 |    1.00 |         — |
| Context Recall    |    1.00 |    1.00 |         — |
| Context Precision |    0.47 |    0.70 |     +0.23 |
| Groundedness      |    0.70 |    0.65 | **-0.05** |
| Correctness       |    1.00 |    1.00 |         — |
| Relevance         |    1.00 |    1.00 |         — |
| Avg Latency       |  17.88s |   7.78s |  **-56%** |
| P95 Latency       |  30.23s |  11.35s |  **-62%** |
| Cost              | $0.7312 | $0.5356 |  **-27%** |

### Result

V2 improved:

* retrieval precision
* context precision
* average latency
* P95 latency
* token usage
* cost

But groundedness decreased:

```text
0.70 → 0.65
```

Therefore:

```text
Quality Gate
     ↓
   FAIL

Regression Gate
     ↓
   FAIL

Release
     ↓
  REVIEW
```

**V2 is not released as-is.**

This demonstrates why RAG optimization should consider multiple quality and operational dimensions rather than relying on a single metric.

See [Final Evaluation](docs/final-evaluation.md).

---

## LLM Judge

Generation evaluation uses a typed judge boundary:

```text
LLM
 ↓
JSON
 ↓
Pydantic Validation
 ↓
GenerationEvaluation
```

LLM output is treated as untrusted data and validated before entering the evaluation pipeline.

The judge abstraction also keeps evaluation independent from a specific LLM provider.

See [Design Decisions](docs/design-decisions.md).

---

## Dataset

The repository contains a small synthetic company-policy dataset designed to exercise different RAG scenarios.

```text
dataset/
├── documents/
├── examples/
├── golden.jsonl
└── golden_v2.jsonl
```

Current evaluation data includes:

* 11 source documents
* 10 golden evaluation cases
* multiple scenario categories
* multi-document cases
* ambiguous queries
* adversarial cases
* permission-sensitive scenarios
* no-answer scenarios

See [Dataset README](dataset/README.md).

---

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
git clone https://github.com/RaVierma/rag-evaluation-harness.git
cd rag-evaluation-harness

uv sync
```

Activate the environment if required:

```bash
source .venv/bin/activate
```

---

## Running Examples

The repository keeps examples focused on the main workflows:

```text
examples/
├── run_rag_evaluation.py
├── run_retrieval_pipeline.py
├── evaluate_retrieval.py
└── regression_check.py
```

Run an example with:

```bash
uv run python examples/run_rag_evaluation.py
```

For retrieval evaluation:

```bash
uv run python examples/evaluate_retrieval.py
```

For regression checking:

```bash
uv run python examples/regression_check.py
```

---

## Testing

Run the complete test suite:

```bash
uv run pytest
```

Verbose:

```bash
uv run pytest -v
```

The test suite covers:

* retrieval metrics
* context metrics
* ranking metrics
* generation evaluation
* judges
* providers
* dataset loading
* evaluation workflow
* aggregation
* comparison
* quality gates
* regression gates
* validation and failure paths

Core evaluation logic is designed to be testable without depending on an external LLM API.

---

## Documentation

| Document                                                 | Purpose                               |
| -------------------------------------------------------- | ------------------------------------- |
| [Architecture](docs/architecture.md)                     | System and package architecture       |
| [Design Decisions](docs/design-decisions.md)             | Important architectural decisions     |
| [Evaluation Methodology](docs/evaluation-methodology.md) | Evaluation process and protocol       |
| [Metrics](docs/metrics.md)                               | Evaluation metrics and dimensions     |
| [Retrieval Baseline](docs/retrieval-baseline.md)         | Retrieval experiments and baseline    |
| [Final Evaluation](docs/final-evaluation.md)             | V1 vs V2 results and release decision |

---

## Engineering Principles

The project follows a simple evaluation loop:

```text
Observation
    ↓
Hypothesis
    ↓
Investigation
    ↓
Change
    ↓
Evaluation
    ↓
Comparison
    ↓
Regression Check
    ↓
Decision
```

Key principles:

* Measure before optimizing
* Evaluate the complete RAG pipeline
* Separate retrieval, context, and generation quality
* Track latency and cost alongside quality
* Compare versions using the same evaluation dataset
* Treat LLM output as untrusted data
* Use explicit quality and regression gates
* Prefer simple, testable architecture over unnecessary abstraction

---

## Project Status

### Completed

* [x] Evaluation dataset
* [x] Retrieval metrics
* [x] Ranking metrics
* [x] Context metrics
* [x] Generation evaluation
* [x] LLM judge abstraction
* [x] Evaluation runner
* [x] Reporting and aggregation
* [x] Version comparison
* [x] Quality gates
* [x] Regression gates
* [x] Retrieval experiments
* [x] RAG runtime structure
* [x] Automated tests

### Next

The evaluation harness provides the foundation for the next stage: building a production-oriented RAG system and evaluating it continuously.

Planned areas include:

* PostgreSQL + pgvector
* production ingestion
* hybrid retrieval
* reranking
* context engineering
* LLMOps
* CI-based regression evaluation
* production observability

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
