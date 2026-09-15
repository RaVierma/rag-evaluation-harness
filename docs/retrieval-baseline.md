# Retrieval Baseline

## Purpose

Establish a measurable retrieval baseline before making further retrieval optimizations.

The baseline evaluates retrieval quality using a fixed corpus, evaluation dataset, and retrieval protocol.

---

## Retrieval Pipeline

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Candidate Retrieval
    ↓
Fusion
    ↓
Reranking
    ↓
Top-K Results
```

The retrieval system can combine dense and sparse retrieval before reranking.

---

## Initial Embedding Experiment

The initial experiment compared two embedding providers while keeping chunking and retrieval unchanged.

### Configuration

* Documents: 11
* Chunks: 27
* Test queries: 5
* Chunk size: 300 characters
* Chunk overlap: 50 characters
* Retriever: `InMemoryRetriever`
* Similarity: cosine similarity
* Top-K: 3 and 5

### Results

| Metric   | Dummy Embedding | Ollama Embedding |
| -------- | --------------: | ---------------: |
| Recall@3 |            0.00 |             0.80 |
| Recall@5 |            0.20 |             1.00 |

### Finding

Semantic embeddings substantially improved retrieval recall compared with the deterministic hash-based embedding.

The experiment established:

```text
Semantic Embedding
       ↓
Cosine Similarity
       ↓
InMemoryRetriever
       ↓
Recall@3 = 0.80
Recall@5 = 1.00
```

The dummy embedding remains useful for deterministic unit tests and provider-independent development.

---

## Day 8 Retrieval Evaluation

The retrieval evaluation was subsequently expanded to a larger evaluation set and a more realistic retrieval pipeline.

### Configuration

* Corpus: 11 documents
* Chunks: 27
* Evaluation cases: 10
* Candidate K: 20
* Final K: 5
* Retrieval: Dense + BM25
* Fusion: Reciprocal Rank Fusion (RRF)
* Reranking: CrossEncoder

### Results

| Metric   | Result |
| -------- | -----: |
| Recall@5 |  1.000 |
| RR@5     |  1.000 |
| NDCG@5   |  0.938 |

### Category Results

| Category       | Score |
| -------------- | ----: |
| Simple         | 1.000 |
| Difficult      | 1.000 |
| Ambiguous      | 0.855 |
| Multi-document | 0.834 |
| Adversarial    | 0.834 |

### Finding

The pipeline retrieved relevant evidence for all evaluated cases.

However, **NDCG@5 = 0.938** shows that relevant evidence was not always ranked optimally.

The most notable issue occurred with authority-sensitive queries, where evidence describing how to request an exception could rank above evidence explicitly describing who can approve it.

---

## Engineering Decision

Do not replace the retrieval or reranking architecture based on this small evaluation set.

Instead:

```text
Current Retrieval
       ↓
Collect Targeted Cases
       ↓
Measure Ranking Behavior
       ↓
Identify Failure Patterns
       ↓
Make One Change
       ↓
Re-evaluate
```

Potential future experiments include:

* BM25 tuning
* hybrid retrieval tuning
* reranking
* metadata filtering
* chunking strategies
* query transformation

Each change should be evaluated against the established baseline.

---

## Key Principle

> **Measure before optimizing.**

A retrieval failure does not automatically mean the retrieval algorithm is incorrect.

Investigation should distinguish between:

1. Representation quality
2. Retrieval quality
3. Ranking quality
4. Chunking quality
5. Query formulation
6. Metadata filtering

The baseline provides the reference point required to evaluate these changes objectively.
