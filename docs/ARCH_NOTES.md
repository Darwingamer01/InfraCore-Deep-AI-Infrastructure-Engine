# InfraCore – Architecture Contract

## Project Purpose
Production-grade AI infrastructure system. Backend/infra only. No UI framework.

## Core Principles
1. Every subsystem has a typed ABC (Abstract Base Class) + Pydantic config
2. All I/O is async (asyncio). No sync blocking calls.
3. Result types are dataclasses, never raw dicts
4. Every subsystem emits Prometheus metrics
5. Config-driven: swap implementations via YAML, not code changes

## Module Interfaces (NEVER break these)

### BaseChunker (src/infracore/chunking/base.py)
- Input: str (raw text), ChunkConfig (Pydantic)
- Output: List[Chunk] – Chunk has: text, start_idx, end_idx, metadata dict

### BaseEmbedder (src/infracore/embedding/base.py)
- Input: List[str] texts, EmbedConfig (Pydantic)
- Output: np.ndarray shape (N, dim)

### BaseVectorStore (src/infracore/vectordb/base.py)
- Methods: upsert(vectors, payloads), search(query_vec, top_k) → List[SearchResult]

### BaseRetriever (src/infracore/retrieval/base.py)
- Input: str query, RetrieverConfig
- Output: List[RetrievalResult] – has chunk, score, metadata

### BaseEvaluator (src/infracore/eval/base.py)
- Input: List[EvalSample] (query, context, answer, ground_truth)
- Output: EvalReport (metrics dict + per-sample scores)

## Stack
- Python 3.11+, FastAPI, Pydantic v2, asyncio
- Qdrant (primary vector store), pgvector (secondary)
- sentence-transformers (BGE-M3, E5-large)
- vLLM / Ollama for inference
- RAGAS + custom probes for eval
- Prometheus + structlog for observability
- pytest-asyncio for all tests

## Sprint Plan
- Sprint 1 (wk 1-2): Ingest + Chunking + Embedding + Qdrant setup
- Sprint 2 (wk 3-4): Hybrid retrieval + Reranking + VectorDB benchmark
- Sprint 3 (wk 5-6): Inference optimization + Agent ReAct loop
- Sprint 4 (wk 7-8): Eval framework CI + Multimodal retrieval

## File Naming
- base.py = ABC + config dataclass
- {name}.py = concrete implementation (e.g. semantic.py, fixed.py)
- test_{name}.py = pytest file, mirrors src structure

## Day 10 — VectorDB Benchmark Findings

**Production Insight**: There is no universal "best" HNSW config. Instead, there exists a **Pareto frontier** where each configuration point represents a different accuracy-latency tradeoff.

### 100K Scale Results (384D)

| Scenario | Config | QPS | Recall@10 | p99ms | Use Case |
|----------|--------|-----|-----------|-------|----------|
| **Speed** | m8_ef64 | 441 | 0.198 | 3.3 | Only viable with hybrid retrieval |
| **Balanced** | m16_ef128 | 254 | 0.682 | 5.2 | Production systems with <10ms SLA |
| **Accuracy** | m32_ef256 | 64 | 0.996 | 23.3 | Batch jobs, high-recall requirements |

**Key Discovery**: At 100K vectors, aggressive `ef` values (64) achieve 441 QPS but find only 20% of true neighbors — unusable for RAG without fallback strategies.

### Dimensionality Impact (10K scale)

Higher vector dimensions (1024D vs 384D) reduce throughput ~40% for the same config, demonstrating the **curse of dimensionality** in approximate search.

### Interview Framing

**Don't say**: "m16_ef128 is best"

**Do say**: "I mapped a Pareto frontier across scale/dimensionality/accuracy. Production choice depends on SLA:
- **<5ms SLA**: ef=64 + hybrid retrieval (accept 80% miss rate)
- **<10ms SLA**: ef=128 (balanced, 254 QPS, 68% recall)
- **Batch jobs**: ef=256 (near-perfect recall, 4× latency)"

**Evidence**: Charts in `eval_reports/qdrant_recall_vs_qps.png` + `qdrant_p99_latency.png` + JSON ground truth validation
