# Final Evaluation Results

## Purpose

Evaluate the RAG system across quality, performance, and cost, then compare V2 against V1 using quality and regression gates.

```text
V1 Baseline
    ↓
V2 Candidate
    ↓
Evaluation
    ↓
Comparison
    ↓
Quality Gate
    ↓
Regression Gate
    ↓
Release Decision
```

---

## Evaluation Results

| Metric            |      V1 |      V2 |    Change |
| ----------------- | ------: | ------: | --------: |
| Recall@K          |    1.00 |    1.00 |      0.00 |
| Precision@K       |    0.58 |    0.77 |     +0.19 |
| MRR               |    1.00 |    1.00 |      0.00 |
| Context Recall    |    1.00 |    1.00 |      0.00 |
| Context Precision |    0.47 |    0.70 |     +0.23 |
| Groundedness      |    0.70 |    0.65 | **-0.05** |
| Correctness       |    1.00 |    1.00 |      0.00 |
| Relevance         |    1.00 |    1.00 |      0.00 |
| Avg Latency       |  17.88s |   7.78s |  **-56%** |
| P95 Latency       |  30.23s |  11.35s |  **-62%** |
| Avg Input Tokens  |   432.4 |   318.9 |      -26% |
| Avg Output Tokens |    55.1 |    38.2 |      -31% |
| Cost              | $0.7312 | $0.5356 |  **-27%** |

---

## Key Findings

### Quality Improvements

V2 improved retrieval and context quality:

```text
Precision@K
0.58 → 0.77

Context Precision
0.47 → 0.70
```

Recall, MRR, context recall, correctness, and relevance remained unchanged.

### Performance Improvements

V2 substantially reduced operational cost:

```text
Average latency
17.88s → 7.78s

P95 latency
30.23s → 11.35s

Cost
$0.7312 → $0.5356
```

This represents approximately **56% lower average latency** and **27% lower evaluation cost**.

### Regression

Groundedness decreased:

```text
0.70 → 0.65
```

This is the primary quality regression identified by the evaluation.

---

## Quality Gate

The quality gate checks whether the candidate satisfies the configured minimum quality requirements.

```text
V2
 ↓
Quality Evaluation
 ↓
FAIL
```

The failure is caused by the groundedness metric falling below the required threshold.

---

## Regression Gate

The regression gate compares V2 against the V1 baseline.

```text
V1: Groundedness = 0.70
V2: Groundedness = 0.65

Regression detected
        ↓
       FAIL
```

The regression gate therefore fails because groundedness degraded beyond the configured tolerance.

---

## Release Decision

```text
Quality Gate
     ↓
   FAIL
     +
Regression Gate
     ↓
   FAIL
     ↓
   REVIEW
```

**Decision: REVIEW — do not release V2 as-is.**

Although V2 provides significant improvements in retrieval precision, context precision, latency, token usage, and cost, the groundedness regression prevents automatic release.

---

## Engineering Conclusion

The experiment demonstrates why RAG optimization should not be evaluated using a single metric.

V2 is:

```text
Better retrieval precision
Better context precision
Lower latency
Lower cost
```

but also:

```text
Worse groundedness
```

Therefore, the correct engineering response is not to automatically accept or reject the optimization based on one improvement.

The evaluation harness makes the trade-off visible and converts it into an explicit release decision.

> **Quality improvement + performance improvement does not automatically mean release. Regression checks must also pass.**
