# InfraCore – Complete Ultra Deep Dive

Welcome to the definitive deep dive into **InfraCore**, a production-grade AI infrastructure engine built entirely from first principles. 

This guide takes you from beginner to advanced concepts in ML/AI systems engineering, mapping theoretical foundations directly to actual production code.

---

## 1. Introduction: Why Build AI Infrastructure?

Most tutorials rely heavily on frameworks like LangChain or LlamaIndex. While these are excellent for rapid prototyping, they obscure the underlying mechanics of AI systems. **InfraCore** strips away the magic. It provides a pure backend system—no UIs, no black-box abstractions—focusing purely on performance, concurrency, and observability.

### The 7 Subsystems of InfraCore
1. **RAG Pipeline**: Semantic, recursive, and late chunking of documents.
2. **VectorDB Benchmarking**: Trade-off analysis between Qdrant, Weaviate, and pgvector.
3. **Inference Optimization**: High-throughput LLM serving via vLLM (PagedAttention, GPTQ).
4. **Agent Orchestration**: A custom ReAct (Reason + Act) loop with a strict typed tool registry.
5. **Evaluation Framework**: Automated scoring using RAGAS and custom faithfulness probes.
6. **VLM + Multimodal Retrieval**: Image/PDF processing using CLIP and LLaVA.
7. **Observability**: Exhaustive Prometheus metrics and traces for every pipeline stage.

---

## 2. Architecture & Core Principles

Before diving into the code, you must understand the architectural contract of InfraCore. 

**Core Principles:**
1. **Abstract Base Classes (ABCs) + Pydantic:** Every subsystem has a base class and a typed `ConfigDict` configuration.
2. **Pure Async I/O:** Every network or disk call is handled via `asyncio`. Blocking operations are strictly forbidden.
3. **Dataclasses for Data Transfer:** Result types are always structured dataclasses, never raw dictionaries.
4. **Observability Built-In:** Every subsystem emits Prometheus metrics (Counters, Histograms) natively.
5. **Configuration-Driven:** Implementations are swapped via YAML configs, not hardcoded logic.

### Standard Module Structure
```text
src/infracore/{subsystem}/
├── base.py          # The ABC and Pydantic configuration model
├── {name}.py        # The concrete implementation (e.g., semantic.py, bge_m3.py)
```

---

## 3. Subsystem 1: Ingestion & Chunking

### The Theory
Large Language Models have finite context windows. To feed them huge documents, we must break the text into manageable "chunks."
- **Fixed Chunking**: Splitting text blindly every N characters. (Fast, but breaks context).
- **Semantic Chunking**: Splitting text intelligently by analyzing sentence boundaries or semantic meaning, ensuring that a single thought isn't cut in half.

### Code Reading Lab: `semantic.py`
In `src/infracore/chunking/semantic.py`, we implement a pure-Python semantic chunker that uses regex to find sentence boundaries, completely avoiding heavy dependencies like NLTK or spaCy.

**Key Concepts in Code:**
- **Regex Boundary Detection:** `re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$", text)` is used to split text into sentences while handling abbreviations like "Dr." or "U.S.A."
- **Token Limits:** The code loops over sentences, accumulating them until adding the next sentence would exceed `self.config.max_tokens`.
- **Overlap:** To preserve context between chunks, the system automatically duplicates the last sentence of the previous chunk into the beginning of the next chunk.

---

## 4. Subsystem 2: Embeddings

### The Theory
An embedding is a numerical representation (a high-dimensional vector) of text semantics. Similar concepts end up physically closer in this vector space.
- **Dimensionality:** BGE-M3 models output 1024-dimensional vectors.
- **Normalization:** By applying L2 normalization, we can compute similarities using the Dot Product, which is significantly faster than computing Cosine Similarity on unnormalized vectors.

### Code Reading Lab: `bge_m3.py`
In `src/infracore/embedding/bge_m3.py`, we use the `sentence-transformers` library to wrap the BAAI/bge-m3 model.

**Key Concepts in Code:**
- **Device Auto-Detection:** The embedder dynamically routes execution to the fastest available hardware (`cuda` > `mps` > `cpu`).
- **Batch Processing:** Processing embeddings one by one is slow. The code batches texts (`batch_size=32`) to saturate GPU cores.
- **Prometheus Metrics:** We define a `Histogram` for latency and a `Counter` for throughput, incrementing them immediately after batch execution.
- **L2 Normalization:** 
  ```python
  norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
  embeddings = embeddings / (norms + 1e-10)
  ```

---

## 5. Subsystem 3: Vector Databases & HNSW

### The Theory
When you have millions of embeddings, comparing a query vector to every single document vector (K-Nearest Neighbors / Exact Search) is computationally unfeasible. Vector databases solve this using **Approximate Nearest Neighbors (ANN)** algorithms, the most popular being **HNSW (Hierarchical Navigable Small World)** graphs.

### The Pareto Frontier (Accuracy vs. Latency)
As noted in `ARCH_NOTES.md`, there is no "perfect" HNSW config. It is always a tradeoff:
- **`ef_construct`**: Determines the quality of the graph built during ingestion (higher = slower insert, better search).
- **`ef_search` (`ef`)**: Determines how many nodes are explored during a query (higher = more accurate, slower query).

**Real-world 100K Scale Benchmarks:**
- **Speed (<5ms SLA):** `ef=64` yields 441 QPS but only 20% recall. (Use only with hybrid/BM25 fallback).
- **Balanced (<10ms SLA):** `ef=128` yields 254 QPS and 68% recall.
- **Accuracy (Batch Jobs):** `ef=256` yields 64 QPS and 99.6% recall.

### Code Reading Lab: `qdrant_store.py`
In `src/infracore/vectordb/qdrant_store.py`, we implement an async wrapper for Qdrant.

**Key Concepts in Code:**
- **Async I/O:** Uses `AsyncQdrantClient` for non-blocking HTTP/gRPC requests.
- **Batch Upserting:** Vectors and payloads are zipped into `PointStruct` objects and pushed to Qdrant in chunks of 100 to prevent network timeouts.
- **Metric Mapping:** `Distance.COSINE`, `Distance.EUCLID`, and `Distance.DOT` are configurable via the `QdrantConfig` Pydantic model.

---

## 6. Subsystem 4: Agent Orchestration (ReAct)

### The Theory
Large Language Models generate text sequentially. They cannot natively "do" things. To give an LLM agency, we use the **ReAct (Reason + Act)** pattern. 
The LLM is prompted to:
1. **Think:** Plan what needs to be done.
2. **Act:** Output a command (e.g., calling a search tool).
3. **Observe:** The system executes the tool and feeds the result back to the LLM.
4. **Repeat** until a "Final Answer" is reached.

### Code Reading Lab: `react_agent.py`
In `src/infracore/agents/react_agent.py`, the core orchestration loop is implemented.

**Key Concepts in Code:**
- **The Execution Loop:** The agent operates inside a `for step_num in range(1, self.config.max_steps + 1):` loop. If it hits the step limit without answering, it aborts to prevent infinite loops (and massive API bills).
- **Tool Registry:** The agent has a `ToolRegistry` containing `BaseTool` implementations. When the LLM outputs an `action_name`, the registry invokes the correct Python function.
- **The Scratchpad:** Every cycle, the Thought, Action, and Observation are appended to a `scratchpad` list. This historical context is injected into the next LLM prompt so the agent remembers what it just did.
- **Robust Error Handling:** If the LLM hallucinates a tool name, the system intercepts it:
  `observation = f"Error: tool '{tool_name}' not found. Available tools: ..."`
  This allows the LLM to realize its mistake and try again on the next iteration.

---

## 7. Next Steps & Extension

InfraCore is designed to be highly modular. By adhering to the ABCs defined in the base files, you can seamlessly extend the system.

**Challenges for the Reader:**
1. **Implement `WeaviateVectorStore`:** Look at `qdrant_store.py`, inherit from `BaseVectorStore`, and implement the `upsert` and `search` methods using the Weaviate v4 Python client.
2. **Create a `LateChunkingEmbedder`:** Extend the embedding system to implement late chunking (embedding the entire document first, then pooling token embeddings over chunk boundaries to preserve global context).
3. **Integrate Reranking:** Add a Cross-Encoder (like BGE-Reranker) between the VectorDB retrieval step and the final LLM generation step to drastically improve context relevance.

### Running the System
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d  # Starts Qdrant and Prometheus
pytest tests/
```

Welcome to production-grade AI infrastructure.
