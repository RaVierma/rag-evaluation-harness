# RAG Evaluation Harness

> **A production-oriented evaluation framework for measuring, comparing, artifacting, and regression-testing RAG systems across retrieval quality, context quality, answer quality, latency, and cost.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/) [![Pydantic](https://img.shields.io/badge/Pydantic-2.x-e92063.svg)](https://docs.pydantic.dev/) [![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

RAG optimization is not a single-metric problem.

A change can improve retrieval while hurting groundedness, or reduce latency and cost while introducing a quality regression.

This project provides an evaluation workflow that makes those trade-offs measurable, reproducible, and reviewable.

```text
RAG System

    ↓

System Output

    ↓

Evaluation

    ↓

Metrics

    ↓

Artifact

    ↓

Comparison

    ↓

Quality Gate + Regression Gate

    ↓

Release Decision
```

The core principle is:

> **Measure before optimizing.**

The evaluation system does not only calculate metrics. Each evaluation run produces a structured **Artifact** containing the evidence required to understand, compare, reproduce, and review the result.

---

## Architecture

The project separates **RAG execution**, **evaluation**, and **artifact generation**.

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

The evaluation layer owns measurement, diagnosis, reporting, artifact creation, comparison, and release gates.

* **`workflow.py`** — coordinates the evaluation workflow.
* **`retrieval.py`** — evaluates retrieval results.
* **`relevance.py`** — evaluates ranking/relevance using relevance judgments.
* **`metrics/`** — individual metric calculations.
* **`diagnostics.py`** — derives RAG-level diagnoses from evaluation signals.
* **`reporting/`** — builds category and evaluation reports.
* **`artifact/`** — preserves structured evaluation results.
* **`comparison.py`** — compares evaluation results across versions.
* **`quality_gate.py`** — evaluates quality thresholds.
* **`regression_gate.py`** — detects regressions against a baseline.
* **`release.py`** — produces the release decision.

### Artifact

The Artifact layer captures the evaluation evidence produced during a run so that results are:

* reproducible
* inspectable
* comparable
* traceable to a system version
* usable by quality and regression gates
* available for later analysis

An evaluation Artifact represents the result of an evaluation run rather than being the RAG runtime itself.

> **RAG executes. Evaluation measures. Diagnostics interpret. Reporting presents. Artifact preserves evidence. Gates decide.**

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

Metrics are calculated during evaluation and become part of the resulting evaluation Artifact.

See [Metrics](docs/metrics.md).

---

## Evaluation Workflow

A fixed golden dataset is used to compare system versions consistently.

```text
Golden Dataset

      ↓

Evaluation Protocol

      ↓

Evaluation Workflow

      ↓

Evaluation Runner

      ↓

Collect Results

      ↓

Measure

      ↓

Create Artifact

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

The Artifact preserves the evaluation result produced by this workflow, allowing a run to be inspected independently of the live execution.

See [Evaluation Methodology](docs/evaluation-methodology.md).

---

## Evaluation Artifact

The Artifact is a first-class output of the evaluation workflow.

Instead of treating evaluation output as temporary console output, the harness produces a structured representation of the evaluation run.

```text
                  Evaluation Run

                       │

         ┌─────────────┼─────────────┐
         ↓             ↓             ↓

      Per-case      Aggregate      Metadata
       Results       Metrics

         │             │             │

         └─────────────┼─────────────┘
                       ↓

                Evaluation Artifact

                       │

              ┌────────┴────────┐
              ↓                 ↓

         Comparison           Gates

              │                 │

              └────────┬────────┘
                       ↓

                Release Decision
```

An Artifact provides a stable evaluation record containing the evidence needed by downstream evaluation components.

This allows the harness to separate:

1. **Execution** — run the evaluation.
2. **Measurement** — calculate evaluation metrics.
3. **Artifact creation** — preserve the evaluation result.
4. **Comparison** — compare artifacts across versions.
5. **Decision** — apply quality and regression gates.

### Why Artifact exists

Without an explicit Artifact boundary, evaluation systems tend to couple:

* metric calculation
* reporting
* comparison
* regression detection
* release decisions

The Artifact provides a stable contract between these stages.

This makes it possible to reason about an evaluation run as a concrete object rather than as transient output.

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

The retrieval evaluation results can be captured as part of an evaluation Artifact for comparison with subsequent retrieval implementations.

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

The evaluation Artifact preserves the evidence behind this decision, allowing the V1 and V2 evaluation results to be compared rather than relying only on the final release status.

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
 ↓
Evaluation Artifact
```

LLM output is treated as untrusted data and validated before entering the evaluation pipeline.

The judge abstraction also keeps evaluation independent from a specific LLM provider.

Judge results become part of the evaluation evidence used to construct the Artifact.

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

Run an evaluation:

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

The evaluation workflow produces the structured Artifact used by downstream comparison and gate evaluation.

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
* evaluation runner
* artifact creation
* artifact validation
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

Artifact

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
* Preserve evaluation results as structured Artifacts
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
* [x] Evaluation workflow
* [x] Retrieval evaluation
* [x] Ranking/relevance evaluation
* [x] RAG diagnostics
* [x] Artifact module
* [x] Structured evaluation artifacts
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
