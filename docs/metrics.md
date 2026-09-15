# Metrics

The evaluation harness measures RAG quality and system performance across four dimensions.

```text
Retrieval
├── Recall@K
├── Precision@K
├── Reciprocal Rank (RR)
└── Mean Reciprocal Rank (MRR)

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

## Metric Groups

### Retrieval

Measures whether relevant information is retrieved.

### Ranking

Measures how effectively relevant results are ordered within the retrieved results.

### Context

Measures the quality and completeness of the context provided to the generation model.

### Generation

Measures the quality of the generated response.

### Performance

Measures operational characteristics such as latency, token usage, and cost.

> **The harness evaluates quality and performance separately so improvements in one dimension do not hide regressions in another.**
