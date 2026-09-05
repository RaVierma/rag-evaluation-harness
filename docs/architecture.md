# Architecture

```
Dataset
   ↓
Evaluation Runner
   ├── Retrieval Metrics
   ├── Context Metrics
   ├── Generation Judge
   └── System Metrics
          ↓
      Aggregation
          ↓
        Report
          ↓
    Quality / Regression
```