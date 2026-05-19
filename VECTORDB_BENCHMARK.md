# VectorDB Benchmark: Production-Grade Insights

## Sprint 2 — Day 10: Qdrant HNSW Tradeoff Analysis

### Executive Summary

Built a **comprehensive benchmarking suite** to measure Qdrant HNSW performance across scale, dimensionality, and configuration. The analysis reveals there is **no universal "best" configuration** — instead, a Pareto frontier where each point represents a different accuracy-latency tradeoff.

### Methodology

- **Dataset**: Gaussian-distributed synthetic vectors (reproducible, ground-truthed via brute-force)
- **Scale**: 10K and 100K vectors
- **Dimensionality**: 384D (embedding size) and 1024D (shows curse of dimensionality)
- **Metric**: Recall@10 (correct top-10 neighbors found / 10), p99 latency (tail latency matters in production)
- **Baseline**: Brute-force cosine similarity (O(N) ground truth, correct by definition)

### Key Findings

#### 1. The Pareto Frontier at 100K Scale

| Config | QPS | Recall@10 | p50ms | p99ms | Production Use Case |
|--------|-----|-----------|-------|-------|---------------------|
| **m8_ef64** | 441 | 0.198 | 2.2 | 3.3 | ❌ Only with hybrid retrieval (accept 80% miss rate) |
| **m16_ef128** | 254 | 0.682 | 3.8 | 5.2 | ✅ Production baseline (<10ms SLA, >50% recall) |
| **m32_ef256** | 64 | 0.996 | 15.4 | 23.3 | ⚠️ Batch/offline jobs (violates typical SLA) |

**Critical Discovery**: At 100K vectors, `ef=64` achieves 441 QPS but finds only 19.8% of true neighbors — incompatible with RAG use cases without reranking/fallback.

#### 2. Dimensionality Impact

```
m16_ef128 on 10K/384D:   632 QPS
m16_ef128 on 10K/1024D:  384 QPS   ← 39% slower
```

Higher dimensions increase ANN search cost — the **curse of dimensionality** in practice. At 1024D, even small datasets show latency increase.

#### 3. SLA Compliance

- **3.3ms p99** (m8_ef64 @ 100K): Suitable for ultra-low-latency systems
- **5.2ms p99** (m16_ef128 @ 100K): Meets typical 10ms RAG SLA
- **23.3ms p99** (m32_ef256 @ 100K): Violates SLA, suitable only for batch

### Visual Evidence

**Chart 1: Recall@10 vs QPS Tradeoff**

Shows the classic ANN frontier — fast/inaccurate (bottom-right) vs slow/accurate (top-left). Points cluster by scale and dimensionality, proving that:
- Small scale (10K) shows near-perfect recall at all configs
- Large scale (100K) exposes recall degradation for aggressive `ef` values
- Dimensionality shifts entire frontier left (lower QPS)

**Chart 2: p99 Latency by Config**

- Green bars (under 10ms SLA): m8_ef64 (3.3ms), m16_ef128 (5.2ms), others
- Red bars (over SLA): m32_ef256 @ 100K (23.3ms) — production red flag

### Interview-Ready Framing

**Instead of:** "m16_ef128 is best"

**Say:**

> I ran a Pareto frontier analysis on HNSW configurations across two dimensions: scale (10K → 100K) and vector dimensionality (384D → 1024D). 
>
> The data shows that recall degrades sharply when you increase scale, especially with aggressive `ef` values. At 100K vectors:
>
> - **ef=64**: 441 QPS but only 19.8% recall — unusable for RAG without hybrid retrieval
> - **ef=128**: 254 QPS with 68% recall — production sweet spot for most SLAs
> - **ef=256**: 64 QPS with 99.6% recall — suitable for high-accuracy batch jobs
>
> The choice is entirely context-dependent. For a **latency-critical system** with a 10ms SLA, I'd use ef=128. For a **batch job** where latency doesn't matter, I'd use ef=256. For a **cost-optimized system** willing to accept lower recall, I'd pair ef=64 with BM25 fallback.

### Production Recommendations

1. **Real-time RAG** (chat, search) → `m16_ef128` (254 QPS, <10ms p99)
2. **Batch analytics** (research, offline) → `m32_ef256` (99.6% recall, latency unconstrained)
3. **Cost-critical** (demo, prototype) → `m8_ef64` + hybrid retrieval (accept miss rate)

### What This Proves

✅ **Infra-level thinking** — Understanding ANN tradeoffs, not just "faster = better"
✅ **Production awareness** — Measuring p99 latency, not just mean
✅ **Scientific rigor** — Ground truth validation, reproducible datasets
✅ **Scale sensitivity** — Showing how results change at 10x scale
✅ **Dimensionality awareness** — Proving curse of dimensionality with data

### Files

- Results: `eval_reports/qdrant_bench_dataset_n100000_d384_gaussian_*.json`
- Charts: `eval_reports/qdrant_recall_vs_qps.png`, `qdrant_p99_latency.png`
- Analysis: `VECTORDB_ANALYSIS.md`

---

## Summary: Pareto Frontier ≠ Best Config

The key insight separating junior and senior engineers is understanding that **optimization is context-dependent**. This benchmark proves the concept with hard numbers.
