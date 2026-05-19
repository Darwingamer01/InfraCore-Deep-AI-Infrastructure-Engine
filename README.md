# InfraCore

Production-grade AI infrastructure engine — no UI, pure backend.

**What this is:** Everything behind an AI application (RAG pipelines, vector search, inference optimization, agent orchestration, evaluation) — built from first principles with zero LangChain shortcuts.

**Why it matters:** Understand *what* AI tools are, not just how to use them. Benchmarks prove it.

## 7 Subsystems

1. **RAG Pipeline** — Fixed, semantic, recursive, late chunking
2. **VectorDB Benchmarking** — Qdrant vs Weaviate vs pgvector
3. **Inference Optimization** — vLLM PagedAttention + GPTQ + speculative decoding
4. **Agent Orchestration** — ReAct loop with typed tool registry
5. **Evaluation Framework** — RAGAS + custom faithfulness probes
6. **VLM + Multimodal Retrieval** — CLIP + LLaVA for images/PDFs
7. **Observability** — Prometheus metrics + Arize Phoenix traces

## Quick Start

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # or: . .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start Local Services

```bash
docker compose up -d
# Verify Qdrant:
curl http://localhost:6333/healthz
# Verify Prometheus:
curl http://localhost:9090/-/healthy
```

### 4. First Build (Sprint 1, Day 1)

```bash
# Review ARCH_NOTES.md first
# Then start with src/infracore/chunking/base.py
python -m pytest tests/ -v
```

## Architecture

Every module follows this pattern:

```
base.py          → ABC + Pydantic config
{implementation}.py → Concrete class
test_{name}.py   → Pytest suite
bench_{name}.py  → Benchmark harness (CLI + comparison)
```

**See [ARCH_NOTES.md](ARCH_NOTES.md) for the complete contract.**

## Stack

- **Python:** 3.11+, asyncio
- **API:** FastAPI + Pydantic v2
- **VectorDB:** Qdrant (primary), pgvector (fallback)
- **Embeddings:** sentence-transformers (BGE-M3, E5-large)
- **Inference:** vLLM, Ollama
- **Eval:** RAGAS + custom probes
- **Observability:** Prometheus, structlog
- **Testing:** pytest-asyncio
- **Config:** YAML-driven, no hardcoded values

## 8-Week Sprint Plan

| Sprint | Weeks | Deliverable |
|--------|-------|-------------|
| 1 | 1–2 | Ingest + Chunking + Embedding + VectorDB setup |
| 2 | 3–4 | Hybrid retrieval + Reranking + VectorDB benchmarks |
| 3 | 5–6 | Inference optimization + Agent ReAct loop |
| 4 | 7–8 | Eval framework CI + Multimodal retrieval |

## Sprint 1 Build Order

| Day | Task | Proves |
|-----|------|--------|
| 1 | `chunking/base.py` → `fixed.py` → `semantic.py` | You own chunking |
| 2 | `embedding/base.py` → `bge_m3.py` | BGE-M3 async inference |
| 3 | `vectordb/base.py` → `qdrant_store.py` | HNSW config |
| 4 | `ingest/base.py` → `pdf_parser.py` | pypdfium2 extraction |
| 5 | Unit tests + `docker compose up` | Qdrant running locally |
| 6–7 | `benchmarks/chunking_bench.py` | Fixed vs semantic MRR@10 |

## Testing

```bash
# All tests
pytest

# Single test file
pytest tests/unit/test_chunking.py -v

# With coverage
pytest --cov=src tests/
```

## Benchmarking

Each subsystem ships with a benchmark harness:

```bash
python benchmarks/chunking_bench.py --config configs/default.yaml --dataset nq
```

Output: JSON report with latency, throughput, accuracy metrics.

## Observability

Metrics available at http://localhost:9090 (Prometheus).

Every pipeline step emits:
- Latency histograms
- Retrieval hit rates
- Cache performance
- Token counts

## License

Internal — not for external use.
