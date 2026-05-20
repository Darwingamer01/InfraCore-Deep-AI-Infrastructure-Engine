# Vector Database Comparison Report

## Results Summary

| Backend | Config | Dataset Size | QPS | Recall@10 | p99 Latency (ms) | RAM Delta (MB) | Upsert Time (s) |
|---------|--------|--------------|-----|-----------|-----------------|----------------|----------------|
| Qdrant | m8_ef64 | 10,000 | 612 | 1.000 | 2.4 | +18 | 1.87 |
| Qdrant | m8_ef128 | 10,000 | 612 | 1.000 | 2.2 | -2 | 1.79 |
| Qdrant | m16_ef128 | 10,000 | 632 | 1.000 | 2.2 | +3 | 1.79 |
| Qdrant | m16_ef256 | 10,000 | 581 | 1.000 | 2.9 | -1 | 1.77 |
| Qdrant | m32_ef256 | 10,000 | 204 | 1.000 | 37.7 | -33 | 1.77 |
| pgvector | pgvector_hnsw_m16_ef128 | 10,000 | 352 | 1.000 | 3.5 | -0 | 17.19 |

## Key Findings

**Best QPS**: Qdrant (632)

**Best Latency**: Qdrant (2.2ms p99)

**Best Recall**: Qdrant (1.000)

**Best RAM Efficiency**: Qdrant (-33MB)

## Analysis

### Throughput

Qdrant: 612 QPS | pgvector: 352 QPS

**pgvector is 42.6% faster**

### Latency

Qdrant: 2.4ms p99 | pgvector: 3.5ms p99

**Qdrant is 43.7% faster**

### RAM Usage

Qdrant: +18MB | pgvector: -0MB

### Summary

- **Qdrant** excels at pure vector search performance (throughput, latency)
- **pgvector** offers tighter Postgres integration for operational simplicity
- Choose **Qdrant** for high-throughput, latency-sensitive workloads
- Choose **pgvector** for mixed SQL+vector queries or existing Postgres infrastructure
