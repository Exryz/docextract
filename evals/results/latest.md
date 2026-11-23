# docextract Evaluation Benchmark Report

**Mode:** Offline Deterministic Benchmark (ScriptedLLM + PDF Layout + Validation Engine)  
**Total Samples Evaluated:** 24 (9 clean/scan, 15 fault-injected)

## Overall Pipeline Metrics

| Metric | Score |
|---|---|
| Field Micro Precision | 100.0% |
| Field Micro Recall | 100.0% |
| Field Micro F1 | 100.0% |
| Field Macro F1 | 100.0% |
| Grand Total Exact-Match | 100.0% |
| Bounding-Box Grounding Coverage | 99.1% |
| Fault-Injection Catch Rate | 100.0% |
| Clean Auto-Approval Rate | 100.0% |
| Mean Pipeline Latency | 14.9 ms |

## Breakdown by Document Layout

| Layout | Samples | Micro F1 | Total Exact-Match | Bbox Grounding | Fault Catch Rate |
|---|---|---|---|---|---|
| `corporate_table` | 8 | 100.0% | 100.0% | 99.1% | 100.0% |
| `modern_sidebar` | 8 | 100.0% | 100.0% | 99.2% | 100.0% |
| `thermal_receipt` | 8 | 100.0% | 100.0% | 98.9% | 100.0% |
