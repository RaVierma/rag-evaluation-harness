# Retrieval Baseline

## Purpose

Establish a measurable baseline for the RAG retrieval pipeline before introducing retrieval optimizations such as reranking, hybrid search, or vector databases.

The experiment compares a deterministic hash-based dummy embedding provider against a semantic embedding provider using Ollama.

The goal is to determine whether semantic embeddings improve retrieval quality while keeping the chunking and retrieval algorithms unchanged.

---

## Retrieval Pipeline

```text
Documents
    ↓
TextDocumentLoader
    ↓
TextChunker
    ↓
EmbeddingProvider
    ↓
EmbeddedChunk
    ↓
InMemoryRetriever
    ↓
Cosine Similarity
    ↓
Top-K Results
```

The experiment changes only the embedding provider.

---

## Experimental Configuration

### Dataset

* Documents: 11
* Generated chunks: 27
* Test queries: 5

### Chunking

* Strategy: structure-aware chunking
* Chunk size: 300 characters
* Overlap: 50 characters
* Chunking implementation: `TextChunker`

### Retrieval

* Retriever: `InMemoryRetriever`
* Similarity: cosine similarity
* Top-K evaluated: 3 and 5
* Search strategy: exhaustive in-memory comparison

### Embedding Providers

#### Baseline

`DummyEmbeddingProvider`

The dummy provider generates deterministic vectors from a hash of the input text. It does not encode semantic meaning.

#### Semantic

`OllamaEmbeddingProvider`

The Ollama provider generates semantic embeddings using a real embedding model.

---

## Test Queries

1. How many vacation days can employees carry forward?
2. What happens to unused vacation days above the carryover limit?
3. Can an employee get an exception to the vacation carryover limit?
4. What are the requirements for working remotely?
5. Who can approve an exception to the annual leave policy?

---

## Results

| Metric   | Dummy Embedding | Ollama Embedding |
| -------- | --------------: | ---------------: |
| Recall@3 |            0.00 |             0.80 |
| Recall@5 |            0.20 |             1.00 |

### Interpretation

The dummy embedding retrieved a relevant chunk within the top 3 for none of the five queries.

The relevant chunk appeared within the top 5 for only one query.

The semantic Ollama embedding substantially improved retrieval:

* Recall@3 increased from **0.00 → 0.80**
* Recall@5 increased from **0.20 → 1.00**

This provides evidence that semantic embeddings are substantially better suited to this retrieval task than the deterministic hash-based representation.

---

## Failure Analysis

The remaining Recall@3 failure occurred for:

> Who can approve an exception to the annual leave policy?

The relevant chunk was:

```text
doc-003-chunk-0001
```

with the content:

> Exceptions may be granted when business circumstances prevented the employee from taking approved leave, subject to manager and HR approval.

With `top_k=5`, the retrieval ranking was:

| Rank | Chunk                |  Score | Relevance |
| ---: | -------------------- | -----: | --------- |
|    1 | `doc-004-chunk-0000` | 0.7793 | No        |
|    2 | `doc-004-chunk-0002` | 0.7427 | No        |
|    3 | `doc-004-chunk-0001` | 0.7304 | Related   |
|    4 | `doc-003-chunk-0000` | 0.7219 | Related   |
|    5 | `doc-003-chunk-0001` | 0.7193 | Yes       |

The relevant chunk was therefore retrieved, but ranked fifth.

### Diagnosis

The embedding model successfully recognized the general topic:

```text
annual leave
exception
carryover
manager
HR
```

However, it did not rank the chunk containing the exact approval relationship above other semantically related chunks.

This indicates that semantic retrieval provides good candidate recall but does not guarantee optimal ranking precision.

---

## Dummy Embedding Failure

The dummy embedding experiment produced highly unrelated results.

For example, the query:

> What happens to unused vacation days above the carryover limit?

returned unrelated chunks about:

* administrator passwords
* remote work
* production deployment
* learning and development

The root cause is that the dummy embedding is based on deterministic hashing rather than semantic meaning.

Therefore:

```text
Text similarity
      ≠
Semantic similarity
```

A deterministic hash can provide reproducibility for tests, but it should not be treated as a meaningful semantic embedding.

---

## Engineering Observation

The experiment demonstrates an important separation of concerns.

Poor retrieval does not automatically mean the retrieval algorithm is incorrect.

The investigation should distinguish between:

1. Representation quality
2. Similarity calculation
3. Candidate retrieval
4. Ranking quality
5. Query formulation
6. Metadata filtering

In this experiment:

* Cosine similarity was mathematically correct.
* The exhaustive in-memory retriever was functioning as designed.
* The dummy representation was inadequate for semantic retrieval.
* Ollama semantic embeddings substantially improved candidate recall.
* Some relevant chunks still received lower rankings than closely related chunks.

---

## Decision

**Decision: Adopt semantic embeddings as the retrieval baseline.**

The deterministic dummy embedding provider remains useful for:

* unit tests
* deterministic testing
* provider-independent development
* testing the retrieval algorithm without external API dependencies

The Ollama semantic embedding provider becomes the baseline for retrieval-quality experiments.

Current baseline:

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

---

## Known Limitation

Recall@5 of 1.00 does not mean the retriever is perfect.

The approval query demonstrates that the correct chunk can be retrieved but ranked too low.

Therefore the next optimization should investigate **ranking quality**, rather than immediately replacing the entire retrieval architecture.

Potential future improvements include:

```text
Dense Retrieval
      ↓
Top-K Candidates
      ↓
Reranking
      ↓
Top-N Context
```

Other future experiments may include:

* BM25
* hybrid dense + sparse retrieval
* reranking
* metadata filtering
* chunking experiments
* query transformation

These should be introduced one variable at a time and evaluated against this baseline.

---

## Engineering Principle

> **Measure before optimizing.**

The retrieval pipeline should follow:

```text
Observation
    ↓
Hypothesis
    ↓
Investigation
    ↓
Root Cause
    ↓
Change
    ↓
Evaluation
    ↓
Comparison
    ↓
Decision
```

The purpose of this baseline is not simply to demonstrate that one embedding provider works better.

It establishes a measurable reference point so future retrieval improvements can be evaluated objectively.
