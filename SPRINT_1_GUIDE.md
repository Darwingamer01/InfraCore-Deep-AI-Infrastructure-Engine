# InfraCore Sprint 1 – Build Guide

## ✅ Sprint 0 Completed: Project Foundation

Your InfraCore environment is fully scaffolded and ready. Everything is set up for Sprint 1 development.

### What's Ready
- ✅ Full project structure with 8 subsystems
- ✅ All abstract base classes (ABCs) defined with Pydantic configs
- ✅ Virtual environment configured (.venv)
- ✅ All dependencies installed (transformers, torch, qdrant-client, etc.)
- ✅ Docker compose for Qdrant + Prometheus
- ✅ pytest configured with asyncio support
- ✅ Configuration templates (default.yaml, eval_ci.yaml)
- ✅ Pydantic v2 compliant with ConfigDict

### Key Files to Know
- **ARCH_NOTES.md** — The architecture contract. Reference it in every new file comment.
- **src/infracore/** — Your module namespace
- **tests/unit/test_imports.py** — Verify module structure (test passes)
- **pyproject.toml** — Full project config
- **docker-compose.yml** — Services: Qdrant (:6333), Prometheus (:9090)

---

## 🚀 Sprint 1 Build Order (7 Days)

### Day 1: Chunking Strategies

**File:** `src/infracore/chunking/fixed.py`

**What to build:** Fixed-size chunker (basic baseline)

**Prompt template:**
```python
# INFRACORE – FixedChunker
# Contract: follows ARCH_NOTES.md BaseChunker interface
# Input: text (str), ChunkConfig with max_tokens=512, overlap=50
# Output: List[Chunk] with text, start_idx, end_idx, metadata
# Rules: no LangChain, async only, pure tokenization by whitespace
# Write the complete implementation:
```

**Key behaviors to test:**
- Chunks text by whitespace tokens
- Respects max_tokens
- Applies overlap correctly
- Each Chunk has correct start_idx, end_idx
- Handles edge cases (empty text, very short text)

**Test framework:**
```python
# tests/unit/test_chunking_fixed.py
import pytest
from src.infracore.chunking.fixed import FixedChunker
from src.infracore.chunking.base import ChunkConfig

@pytest.mark.asyncio
async def test_fixed_chunker():
    config = ChunkConfig(strategy="fixed", max_tokens=10, overlap=2)
    chunker = FixedChunker(config)
    chunks = await chunker.chunk("This is a sample text")
    assert len(chunks) > 0
    assert all(c.start_idx < c.end_idx for c in chunks)
```

---

### Day 2: Semantic Chunking

**File:** `src/infracore/chunking/semantic.py`

**What to build:** Semantic similarity-based chunker

**Strategy:**
- Split text into candidate segments
- Embed each segment with BAAI/bge-m3
- Measure similarity between adjacent segments
- Split where similarity drops below threshold

**Key constraint:** Reuse embedding logic (will build full embedder Day 2)

---

### Day 3: Embeddings – BGE-M3

**File:** `src/infracore/embedding/bge_m3.py`

**What to build:** BGE-M3 embedder (batch async inference)

**Constraints:**
- Batch by EmbedConfig.batch_size
- Return np.ndarray shape (N, 1024)
- Use sentence-transformers library
- Handle device placement (CPU/GPU)

**Test:** Verify embedding shape and consistency

---

### Day 4: VectorDB Setup – Qdrant

**File:** `src/infracore/vectordb/qdrant_store.py`

**What to build:** Qdrant client wrapper

**Methods:**
- `upsert(vectors, payloads, ids)` — async
- `search(query_vector, top_k)` — async search with HNSW config

**Key config:**
```yaml
vectordb:
  store_type: qdrant
  collection_name: infracore_default
  vector_size: 1024
  distance_metric: cosine
```

**Start Qdrant:**
```bash
docker compose up -d qdrant
curl http://localhost:6333/healthz
```

---

### Day 5: Document Ingestion – PDFs

**Files:**
- `src/infracore/ingest/pdf_parser.py` — pypdfium2 extraction
- `src/infracore/ingest/html_parser.py` — simple HTML stripping

**What to build:**
- Read PDF → extract text + metadata (pages, author)
- Preserve table structure where possible
- Return IngestedDocument with source, format, metadata

---

### Days 6–7: Unit Tests + First Benchmark

**Tests to write:**
- `tests/unit/test_chunking_fixed.py` — FixedChunker
- `tests/unit/test_chunking_semantic.py` — SemanticChunker
- `tests/unit/test_embedding_bge.py` — BGE-M3
- `tests/unit/test_vectordb_qdrant.py` — Qdrant upsert/search
- `tests/unit/test_ingest_pdf.py` — PDF parsing

**Run all tests:**
```bash
source .venv/bin/activate
pytest tests/unit/ -v
```

**First Benchmark: Chunking**

**File:** `benchmarks/chunking_bench.py`

**What to measure:**
- Fixed vs Semantic vs Recursive (if time permits)
- Metric: chunks per second (throughput)
- Metric: MRR@10 on NQ dev set (quality)
- Metric: avg chunk size (compliance with config)

**Output:**
```json
{
  "fixed": {
    "throughput_cps": 145.3,
    "avg_chunk_size": 510,
    "dataset": "nq_dev_100"
  },
  "semantic": {
    "throughput_cps": 28.5,
    "avg_chunk_size": 487,
    "dataset": "nq_dev_100"
  }
}
```

**Run:**
```bash
python benchmarks/chunking_bench.py --config configs/default.yaml --dataset nq
```

---

## 📋 Day 1 Immediate Next Steps

1. **Start with `src/infracore/chunking/fixed.py`**
   - Copy the Copilot prompt above into a new file comment
   - Let Copilot generate the class
   - Review against ARCH_NOTES.md
   - Verify it follows the BaseChunker ABC

2. **Write `tests/unit/test_chunking_fixed.py`**
   - Test chunk creation, sizes, indices
   - Test overlap behavior
   - Test edge cases

3. **Run tests:**
   ```bash
   pytest tests/unit/test_chunking_fixed.py -v
   ```

4. **Come back and report:**
   - Paste the generated fixed.py code
   - Report test results
   - Next file: semantic.py

---

## 💡 Copilot Prompting Rules

Every new file starts with a comment header:

```python
# INFRACORE – [ModuleName]
# Contract: follows ARCH_NOTES.md interface for [BaseClass]
# Input: [exact types from ARCH_NOTES]
# Output: [exact types from ARCH_NOTES]
# Rules: [key constraints – no LangChain, async-only, etc.]
# Write the complete implementation:
```

Then hit Enter — Copilot will fill in the entire class.

**Before accepting, verify:**
- ✅ Inherits from correct ABC
- ✅ All abstract methods implemented
- ✅ Pydantic config matches ARCH_NOTES
- ✅ Full type hints
- ✅ All I/O is async
- ✅ No LangChain imports

If Copilot hallucinates, paste the ARCH_NOTES section into chat and say: *"Use this interface exactly."*

---

## 🛠️ Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest tests/unit/ -v
pytest tests/unit/test_chunking_fixed.py -v

# Start services
docker compose up -d

# Check Qdrant health
curl http://localhost:6333/healthz

# Verify module imports
python -c "from src.infracore.chunking.fixed import FixedChunker; print('✓')"

# Run benchmark
python benchmarks/chunking_bench.py --config configs/default.yaml --dataset nq
```

---

## 📊 Success Metrics for Sprint 1

| Deliverable | Metric |
|-------------|--------|
| FixedChunker | 150+ chunks/sec throughput |
| SemanticChunker | 25+ chunks/sec throughput |
| BGE-M3 Embedder | 500+ texts/sec batched |
| Qdrant Setup | <100ms search latency |
| PDF Ingestion | 95%+ text recovery vs ground truth |
| Test Coverage | 80%+ of chunking module |
| Benchmark Report | JSON with fix/semantic comparison |

---

## 🚨 If Copilot Hallucinates

1. **Too many imports from LangChain?**
   - Paste ARCH_NOTES.md into chat
   - Say: "This is the contract. Use these ABCs only."

2. **Async syntax wrong?**
   - Show it your test file as reference
   - Ask: "Make all I/O async like this test"

3. **Config doesn't match?**
   - Paste the specific Config from ARCH_NOTES
   - Say: "This is the exact signature"

---

## 🎯 Philosophy

Every file you write proves something:

- **FixedChunker** → You understand tokenization, not LangChain
- **SemanticChunker** → You understand semantic similarity, embedding strategies
- **BGE-M3** → You understand batch inference, device placement, model loading
- **Qdrant** → You understand vector indices, similarity search, HNSW tuning
- **Benchmarks** → You understand that quality is measured, not guessed

By day 7, you'll have 3 benchmarks proving:
- Fixed chunking ≈ 5× faster than semantic, but lower quality
- Semantic chunks produce better retrieval at the cost of latency
- Every tradeoff is measurable

---

Start with Day 1 now. Report back with fixed.py when ready. 🚀
