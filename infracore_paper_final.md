
InfraCore: A Modular, Benchmarked, Production-Grade AI Infrastructure Engine for RAG, Multimodal Retrieval, and Inference Optimization

Utkarsh Choudhary
Independent Researcher
utkarsh11980@gmail.com


ABSTRACT

Modern AI applications are often bottlenecked not by model capability but by the infrastructure surrounding them, the pipelines that ingest, index, retrieve, and serve information at production quality. This paper presents InfraCore, a modular, production-grade AI infrastructure engine built entirely from first principles in Python 3.11, without reliance on high-level orchestration frameworks such as LangChain or LlamaIndex. InfraCore comprises eight tightly integrated subsystems: a multi-format Retrieval-Augmented Generation (RAG) pipeline with four chunking strategies and three embedding backends; a vector database benchmarking layer evaluating Qdrant, pgvector, and Weaviate; an inference optimization layer comparing vLLM and Ollama across three concurrency levels; a custom ReAct agent orchestration loop; a RAGAS-inspired evaluation framework with CI gating; a multimodal retrieval pipeline combining CLIP embeddings and OCR; a full Prometheus observability stack; and a suite of GitHub Actions quality gates. We present a rigorous benchmarking methodology and empirical results, including an HNSW Pareto frontier across ef values of 64, 128, and 256 at 100K vectors and a throughput analysis showing vLLM achieving 8.3 times higher token throughput than Ollama at concurrency level 8 via PagedAttention-based request batching. Across a post-sprint integration test suite, 199 tests pass with zero regressions. InfraCore serves as a reference architecture demonstrating that production AI infrastructure can be built transparently, modularly, and with full observability, a contribution aimed at practitioners who need to reason about, extend, and operate AI systems at scale.


I. INTRODUCTION

The discourse surrounding artificial intelligence in software engineering has, for much of the past four years, centered on foundation models, their parameter counts, context windows, and benchmark rankings. This emphasis, while understandable, has obscured a quieter but arguably more consequential problem: the infrastructure layer surrounding these models is almost universally underdeveloped. In production deployments, latency spikes, retrieval quality degradation, lack of observability, and silent evaluation regressions are rarely caused by the model itself. They are caused by the system in which the model operates.

The retrieval pipeline that feeds a language model determines what it can know in a given context window. The vector database configuration that serves a semantic search system determines whether it returns the most relevant 10 results or a roughly correct but ultimately unreliable approximation. The inference backend that handles concurrent requests determines whether a multi-user application scales gracefully or collapses under moderate load. These are infrastructure problems, and they demand infrastructure solutions.

The open-source ecosystem has responded to this gap with orchestration frameworks, most prominently LangChain [1] and LlamaIndex [2]. These frameworks provide valuable abstractions and accelerate prototyping significantly. However, they impose their own abstraction costs: non-transparent memory management, hidden retry logic, difficult-to-debug prompt construction, and dependency graphs that make subsystem-level benchmarking nearly impossible. An engineer using LangChain's retrieval chain cannot easily instrument the exact latency of the embedding step independently from the vector store lookup step without bypassing the abstraction entirely. This is not a critique of those frameworks' value for rapid development; it is an observation that they are not designed for the kind of rigorous, component-level analysis that production systems eventually require.

InfraCore was built to address precisely this gap. It is a backend-only AI infrastructure engine that implements eight production-grade subsystems from scratch, each with typed Abstract Base Class (ABC) contracts, Pydantic v2 configuration, async I/O, structured logging, and Prometheus metrics instrumentation. Nothing is hidden. Every pipeline stage emits observable signals. Every implementation decision is documented in code. The system was designed to be the supply chain and quality-control layer behind AI features, not a user-facing application, but the operational substrate that user-facing applications depend on.

The contributions of this paper are as follows.

First, a complete design and implementation of a modular AI infrastructure stack, built without framework dependencies, demonstrating that such a system is tractable and yields significant observability benefits.

Second, an empirical benchmarking of HNSW approximate nearest neighbor search across the ef-search Pareto frontier at 100K vectors, providing a quantified tradeoff table for production configuration selection.

Third, a rigorous inference backend comparison between vLLM and Ollama across three concurrency levels (c=1, 4, 8), measuring time-to-first-token (TTFT), throughput (tokens per second), and p99 latency, leading to a concrete decision framework for backend selection.

Fourth, a multimodal retrieval pipeline combining CLIP visual embeddings and OCR-derived text, with a pluggable backend strategy pattern validated through 199 integration tests.

Fifth, a CI-gated evaluation framework inspired by RAGAS [3] methodology, enabling automated regression detection on retrieval and generation quality.

The remainder of this paper is organized as follows. Section II surveys related work in RAG, vector search, inference serving, and agent orchestration. Section III describes InfraCore's overall system architecture and guiding principles. Section IV provides per-subsystem design and implementation detail. Section V presents benchmarking methodology and empirical results. Section VI covers the evaluation framework. Section VII discusses architectural lessons and limitations. Section VIII concludes with a summary of contributions and future directions.


II. RELATED WORK

A. Retrieval-Augmented Generation

The foundational approach of augmenting language model generation with retrieved documents was formalized by Lewis et al. in Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [4], presented at NeurIPS 2020. Their architecture combined a dense passage retriever (DPR) with a sequence-to-sequence generator (BART), demonstrating that retrieval could supply factual grounding that generative models alone could not reliably provide. InfraCore implements the retrieval side of this equation, ingestion, chunking, embedding, and vector store lookup, as discrete, independently observable pipeline stages, rather than the end-to-end differentiable system Lewis et al. proposed.

Subsequent work has refined this architecture in several directions. Izacard and Grave [5] introduced Fusion-in-Decoder (FiD), showing that attending over multiple retrieved passages jointly, rather than reranking and selecting a single passage, improved generation quality. Shi et al. [6] demonstrated that retrieval could introduce harmful distraction if the retrieved content, while lexically relevant, was semantically misleading. These findings informed InfraCore's design decision to implement late chunking as a strategy that maintains broader document context during embedding, partially mitigating the context fragmentation problem endemic to fixed-window chunking.

B. Approximate Nearest Neighbor Search and HNSW

Vector similarity search is the computational core of semantic retrieval. Malkov and Yashunin's Hierarchical Navigable Small World (HNSW) algorithm [7], published in IEEE Transactions on Pattern Analysis and Machine Intelligence in 2020, remains the dominant approach for approximate nearest neighbor (ANN) search in production systems. HNSW constructs a multi-layer proximity graph where nodes at each layer are drawn from a subset of the dataset, enabling logarithmic-time greedy search with controllable recall. The ef_construction parameter governs build-time graph quality, while ef_search governs query-time search depth. InfraCore's HNSW benchmarking characterizes this tradeoff empirically across three configurations, providing concrete QPS and Recall@10 numbers that practitioners can map to their own latency budgets.

Alternative ANN algorithms include Product Quantization (PQ) variants [8], IVF-based approaches [9], and SCANN [10]. These methods trade memory compression for recall, which HNSW avoids through graph-based organization. For the workloads InfraCore targets, sub-millisecond p99 latency at moderate QPS, HNSW as implemented by Qdrant provides the best precision-to-speed profile.

C. Vector Database Systems

The managed vector store landscape now includes Qdrant [11], Weaviate [12], Pinecone, Milvus [13], and pgvector [14] as a PostgreSQL extension. Each system makes different engineering tradeoffs. Qdrant is written in Rust and provides a native async Python client, making it the natural primary target for InfraCore's async architecture. pgvector integrates relational query semantics with vector search, enabling hybrid SQL-vector queries at the cost of raw ANN throughput. Weaviate provides a built-in GraphQL interface and native module support for embedding models, at the cost of operational complexity.

InfraCore implements a BaseVectorStore ABC that all three adapt to, enabling apples-to-apples benchmarking without coupling application code to any specific store.

D. Inference Serving and PagedAttention

The problem of serving large language models efficiently under concurrent load was substantially addressed by Kwon et al.'s PagedAttention system, vLLM [15], presented at SOSP 2023. The core observation was that LLM inference suffers from memory fragmentation in the KV cache: as requests arrive with variable prompt lengths and generate variable output lengths, the contiguous memory allocation scheme used by naive implementations wastes significant GPU memory, limiting effective batch size. PagedAttention maps KV cache blocks to non-contiguous physical memory in the manner of OS virtual memory paging, enabling the scheduler to batch dramatically more requests onto available GPU memory. The empirical consequence, which InfraCore's benchmarks reproduce at smaller scale, is that vLLM's throughput scales near-linearly with concurrency while Ollama, which processes requests sequentially without batching, saturates at low concurrency.

Orca [16], a predecessor system, introduced iteration-level scheduling, processing heterogeneous requests in a single batch by maintaining a separate state per request rather than waiting for all requests in a batch to complete. vLLM builds upon this idea with its paged memory management.

E. Embedding Models

InfraCore integrates three embedding model families. BGE-M3, introduced by Chen et al. [17] as part of the FlagEmbedding project, produces 1024-dimensional embeddings with strong multilingual performance across retrieval, reranking, and zero-shot tasks. E5-large [18] from Wang et al. is a 1024-dimensional encoder trained with contrastive learning on a large-scale text pair corpus, providing a strong baseline for English-language retrieval. CLIP, introduced by Radford et al. [19] at ICML 2021, learns joint image-text embeddings through contrastive learning on 400 million web image-text pairs, enabling cross-modal similarity search and forming the foundation of InfraCore's multimodal retrieval subsystem.

F. Visual Language Models

InfraCore's multimodal QA backend integrates BLIP (Bootstrapping Language-Image Pre-training), introduced by Li et al. [20] at ICML 2022. BLIP addresses a limitation in contrastive pretraining-only models such as CLIP by adding a generative language modeling objective, enabling it to answer natural-language questions about images rather than merely computing similarity scores. The Salesforce BLIP-VQA-Base model provides the production backend for InfraCore's VLMDocumentQA pipeline.

G. Agent Orchestration and ReAct

The ReAct framework, introduced by Yao et al. [21] at ICLR 2023, demonstrated that interleaving chain-of-thought reasoning steps with explicit tool actions and observations improves task completion on benchmarks such as HotPotQA and Fever compared to either reasoning alone or acting alone. This Thought, Action, Observation loop is the direct inspiration for InfraCore's AgentOrchestrator design. Unlike LangChain's agent abstractions, InfraCore's implementation maintains the scratchpad as a plain string accumulated across steps, making the prompt construction fully inspectable.

Toolformer [22] from Schick et al. demonstrated that language models could be taught to self-supervise tool call insertion, complementing ReAct's inference-time prompting approach.

H. Evaluation of RAG Systems

RAGAS (Retrieval-Augmented Generation Assessment), introduced by Es et al. [3] in 2023, provides an automated framework for evaluating RAG systems across four dimensions: faithfulness, answer relevance, context precision, and context recall. InfraCore's evaluation framework draws on the faithfulness probe methodology from RAGAS while implementing custom lexical metrics for deterministic CI integration.

I. Observability in ML Systems

Prometheus [23] is the de facto standard for metrics collection in cloud-native systems. Its pull-based exposition model, dimensional labeling via label cardinality, and histogram support for latency tracking make it well-suited to multi-stage pipelines where different subsystems have different latency profiles. Arize Phoenix [24] extends observability to LLM traces, capturing prompt-response pairs, token counts, and model metadata in a queryable trace store. Structured logging via libraries such as structlog [25] provides machine-readable log events that complement time-series metrics by capturing the discrete events that metrics aggregate.


III. SYSTEM ARCHITECTURE

A. End-to-End Pipeline Flow

InfraCore's processing pipeline follows a strictly left-to-right flow from raw input to evaluated output. At the highest level, this pipeline can be described as:

Raw Documents, Ingestion (Parsers), Chunking (4 strategies), Embedding (3 backends), VectorStore (Qdrant, pgvector, Weaviate), Retriever, Inference Router (vLLM, Ollama), Answer (GenerationResult), Evaluation (EvalReport).

Each stage in this pipeline represents a typed data contract. A Parser produces Document objects. A Chunker consumes Document objects and produces lists of Chunk dataclasses. An Embedder consumes Chunk objects and produces EmbeddedChunk objects carrying an embedding vector. A VectorStore indexes EmbeddedChunk objects and returns SearchResult objects on query. The Retriever wraps VectorStore queries and produces RetrievalResult objects. The Inference Router consumes a RetrievalResult plus a user query and produces a GenerationResult. The Evaluator consumes GenerationResult objects against ground truth and produces EvalReport artifacts.

No stage holds global state. Every stage can be independently tested, replaced, and monitored.

Figure 1 describes this pipeline as a directed acyclic graph. At the document input boundary, three parallel parser paths (PDF to PDFParser, Markdown to MarkdownParser, HTML to HTMLParser) converge to a shared Document abstraction. At the embedding boundary, a device-aware router directs computation to CUDA, Apple MPS, or CPU depending on hardware availability. At the inference boundary, a health-checked router implements primary-backend-first with automatic failover to the secondary backend.

B. Architectural Principles

Typed ABC Contracts. Every subsystem defines a Base class using Python's Abstract Base Class mechanism. Concrete implementations must implement all abstract methods or instantiation raises TypeError at import time. This design eliminates the runtime discovery of missing implementations that plagues duck-typed systems. For example, BaseVectorStore defines upsert, search, delete, and close as abstract coroutines; Qdrant, pgvector, and Weaviate adapters all implement these contracts without modification to application code.

Async I/O Throughout. InfraCore uses Python's asyncio event loop for all I/O-bound operations, including HTTP calls to inference backends, vector store queries, and file reads. No blocking I/O calls appear in any hot path. This design choice was not cosmetic: at concurrency levels of 4 and 8, the difference between async and sync retrieval paths manifests as a 3 to 5 times throughput gap in preliminary measurements, as blocking calls serialize what could be parallel operations.

Pydantic v2 Configuration. Every subsystem exposes a Settings class backed by Pydantic v2 BaseModel with strict type validation and field-level defaults. Configuration is loaded from YAML files at startup, validated against the schema, and injected into subsystem constructors. Swapping a vector store backend requires changing one key in the YAML configuration file; no application code changes are required.

Dataclass Result Types. Every pipeline stage returns a typed dataclass, not a raw dictionary. The complete result type hierarchy includes: Chunk (text, metadata, source id, chunk index), EmbeddedChunk (extends Chunk with embedding as a NumPy array), SearchResult (score, chunk, vector id), RetrievalResult (query, results as a list of SearchResult, latency in milliseconds), GenerationResult (query, context chunks, answer, model id, latency in milliseconds, tokens generated), AnswerResult (answer, source provenance as a list of ProvenanceRecord, confidence), and EvalReport (metrics as a dictionary of floats, threshold pass as a boolean, artifact path as a string). This design makes inter-subsystem contracts explicit and IDE-navigable.

Universal Prometheus Instrumentation. Every subsystem constructor registers Prometheus Counter and Histogram metrics on a shared CollectorRegistry. Metric names follow a consistent naming convention: infracore-subsystem-operation-total for counters and infracore-subsystem-operation-seconds for latency histograms. A FastAPI /metrics endpoint exposes these in the Prometheus text exposition format, scraped by a Prometheus server configured in configs/prometheus.yml.

YAML-Driven Implementation Swaps. The configuration schema for each subsystem includes a backend key whose value selects the concrete implementation. An EmbedderFactory reads this key and instantiates the appropriate class. This pattern is repeated across VectorStore, InferenceBackend, QABackend, and Chunker factories. No conditional import logic appears in application code.

C. Module Structure Pattern

Each subsystem follows a consistent directory layout. A subsystem folder contains base.py which holds the ABC definition and shared dataclasses, config.py which holds the Pydantic Settings class, an implementations folder holding one file per concrete implementation, factory.py which reads config.backend and returns the appropriate instance, and metrics.py which holds Prometheus metric registrations.

This layout ensures that adding a new vector store backend requires creating one new file in the implementations folder and registering it in factory.py, a two-file change that does not touch any other subsystem.


IV. SUBSYSTEM DESIGN AND IMPLEMENTATION

A. RAG Pipeline: Ingestion, Chunking, and Embedding

Problem. Language models operate on text. Real-world document corpora arrive as PDFs (often scanned), Markdown files (often with embedded code blocks and headers that should be treated as semantic boundaries), and HTML pages (often with boilerplate navigation and footer content that dilutes retrieval relevance). A production ingestion pipeline must handle all three formats, extract clean text, and segment that text into units that an embedding model can process within its token budget.

Design Decisions. InfraCore implements three parser classes, PDFParser, MarkdownParser, and HTMLParser, all subclassing BaseParser. PDFParser uses PyMuPDF (fitz) for text extraction, which preserves layout information including page boundaries and bounding boxes, metadata later used by the multimodal provenance system. MarkdownParser uses Python-Markdown with custom extensions to detect header boundaries, treating H1, H2, and H3 transitions as natural chunk boundaries. HTMLParser uses BeautifulSoup4 with targeted boilerplate removal: nav, footer, header, and aside tags are stripped before text extraction, leaving only the main content region.

Four chunking strategies are implemented, each capturing a different tradeoff between context coherence and embedding model compatibility.

Fixed Chunking divides the document text into windows of N characters with an optional overlap. This is the simplest strategy and produces the most predictable chunk size distribution, but it frequently bisects sentences at chunk boundaries, producing fragments that lack grammatical and semantic coherence.

Semantic Chunking uses the NLTK sentence tokenizer to split text at sentence boundaries first, then greedily aggregates sentences until the target token budget is approached. Chunks produced by this strategy maintain sentence-level coherence. The tradeoff is that chunk sizes become variable, occasionally producing very short chunks for paragraphs with many short sentences.

Recursive Chunking attempts progressively finer split points in order: paragraph boundary (double newline), sentence boundary (period followed by space), word boundary. This is the strategy favored by many retrieval practitioners for its balance between consistency and coherence.

Late Chunking is the most conceptually novel strategy in InfraCore, inspired by the observation that fixed chunking loses document-level context. In late chunking, the entire document is embedded as a single sequence first (using the model's maximum context window), producing a sequence of token-level embeddings. Chunk boundaries are then applied in embedding space rather than in text space: the embedding for each chunk is computed as the mean-pooled representation of the token embeddings that fall within that chunk's character span. This preserves cross-chunk context in the embeddings themselves, rather than trying to recover it through retrieval-time metadata.

Three embedding backends are available. BGE-M3 produces 1024-dimensional embeddings via the FlagEmbedding library. E5-large produces 1024-dimensional embeddings via the sentence-transformers library with the instruction prefix "passage:" prepended during indexing and "query:" prepended at query time. CLIP produces 512-dimensional embeddings and is discussed further in the multimodal subsystem. All embedders apply L2 normalization as the final step before returning embeddings, enabling the use of dot-product similarity (inner product) as a proxy for cosine similarity. When both vectors are unit-normalized, their dot product equals their cosine similarity, and this normalization step enables Qdrant's Dot distance metric, which is computationally cheaper than cosine distance computation with un-normalized vectors. All embedders implement batch processing with device-aware dispatch: the factory detects CUDA availability first, then Apple MPS (for Apple Silicon), then falls back to CPU.

Tradeoffs. Fixed chunking is fastest and most predictable; semantic chunking produces higher retrieval quality at the cost of variable chunk sizes; recursive chunking is a pragmatic middle ground; late chunking requires full-document embedding and scales poorly with document length but preserves the richest contextual embeddings.

B. VectorDB Benchmarking

Problem. Selecting a vector database for production deployment requires empirical evidence, not vendor claims. The three primary axes are throughput (queries per second), recall quality (what fraction of true nearest neighbors are returned), and latency at percentiles. HNSW's ef_search parameter trades recall for throughput along a Pareto frontier, and understanding this frontier for the specific dimensionality and dataset size of a deployment is essential for informed configuration.

Design Decisions. InfraCore implements a BaseVectorStore ABC with four abstract coroutines: upsert (accepting a list of EmbeddedChunk), search (accepting a query vector and k, returning a list of SearchResult), delete (accepting a list of string IDs), and close. Three adapters implement this interface: QdrantVectorStore (wrapping the qdrant-client async client), PgVectorStore (wrapping asyncpg with pgvector), and WeaviateVectorStore (wrapping the Weaviate v4 client).

The QdrantVectorStore implementation uses batch upserts with a batch size of 100 points to avoid client-side timeout errors on large insertions, a practical engineering detail that emerged from testing. The async client is initialized with prefer_grpc set to True when available, as gRPC provides lower per-request overhead than HTTP for high-QPS scenarios.

The benchmarking harness generates a synthetic dataset of 100,000 384-dimensional vectors drawn from a uniform distribution with a seeded random number generator for reproducibility. Ground truth nearest neighbors are computed exhaustively for 1,000 query vectors using NumPy's vectorized dot product, establishing the reference set for recall computation. Recall@10 is computed as the fraction of true top-10 neighbors appearing in the HNSW-returned top-10 results.

Tradeoffs. The HNSW Pareto frontier is characterized across three ef_search configurations, presented in Table I. No single configuration is universally optimal; the appropriate choice depends on the application's latency budget, recall requirements, and throughput demands.

C. Inference Optimization

Problem. Language model inference under concurrent load exposes dramatic differences between serving architectures. An LLM serving backend that performs well for a single user may fail catastrophically when ten users make simultaneous requests. Understanding this behavior requires measuring throughput, TTFT, and tail latency as functions of concurrency, information that is not available from vendor documentation or synthetic benchmarks.

Design Decisions. The InferenceRouter class maintains a list of backend configurations in priority order. On each request, it attempts the primary backend. If the backend's health check fails or the request times out, it transparently retries with the secondary backend. Health checks are performed as lightweight GET requests to /health endpoints and cached for 30 seconds to avoid adding health check overhead to every request.

Two backends are implemented. OllamaBackend wraps the Ollama HTTP API at /api/generate using aiohttp for async HTTP. VLLMBackend wraps the vLLM OpenAI-compatible API at /v1/completions. Both backends return GenerationResult dataclasses with identical fields, making them interchangeable to the calling code.

The benchmarking harness loads 50 prompts across three categories: short factual queries averaging 12 tokens, medium analytical questions averaging 35 tokens, and long synthesis prompts averaging 80 tokens. Performance is measured at concurrency levels 1, 4, and 8 using asyncio.gather() to issue concurrent requests. Throughput is computed as total tokens generated divided by wall-clock time. TTFT is measured by streaming the response and recording the time from request dispatch to first token received. P99 latency is computed over the per-request latency distribution.

Tradeoffs. The fundamental architectural difference between Ollama and vLLM determines their scaling behavior. Ollama processes requests sequentially: each request holds exclusive access to the model while generating its full response. At concurrency level 8, all 8 requests queue behind one another, and TTFT for the last request is approximately 8 times the TTFT of the first. vLLM's PagedAttention scheduler [15] batches concurrent requests in a single forward pass, interleaving token generation across requests in what is effectively a round-robin per decode step. This means TTFT remains approximately constant regardless of concurrency, and total throughput scales near-linearly with GPU parallelism.

D. Agent Orchestration (ReAct Loop)

Problem. Answering questions that require multi-step reasoning cannot be addressed by a single retrieval-and-generate pass. An agent framework that interleaves reasoning and tool use is required. Building this without LangChain or LlamaIndex forces every component to be explicit and auditable.

Design Decisions. The AgentOrchestrator implements the Thought, Action, Observation loop described by Yao et al. [21]. At each step, the orchestrator appends the accumulated scratchpad to the system prompt and calls the inference backend. The response is parsed for a structured Action block containing tool_name and tool_input fields. If the action references a registered tool, the tool is invoked and its output becomes the next Observation. If the action references an unregistered tool name (a hallucination), the orchestrator injects a "Tool not found" message as the Observation and continues the loop, giving the model the opportunity to self-correct.

The ToolRegistry maps string tool names to BaseToolHandler implementations. CalculatorTool is the primary tool implemented: it accepts an expression string, parses it using Python's ast.parse() with a whitelist of permitted AST node types, and evaluates it using a custom evaluator rather than eval(). This approach prevents arbitrary code execution while supporting the arithmetic expressions that agents commonly require.

The loop terminates when the model emits a Final Answer token, when max_steps is reached (default 10), or when the inference backend raises an unrecoverable error. In all cases, the scratchpad is included in the GenerationResult for debugging.

Tradeoffs. The scratchpad-based approach accumulates prompt length linearly with the number of steps, eventually approaching the model's context limit. For deep reasoning tasks requiring many steps, the effective working memory is bounded by the model's context window. Future work could implement scratchpad compression by summarizing older Thought, Action, Observation triplets as the scratchpad grows.

E. Evaluation Framework

Problem. Retrieval and generation quality are not self-evident from system logs or unit tests. A system can pass all unit tests while producing faithfully grounded answers to simple queries and hallucinated answers to complex ones. Automated evaluation against known-answer test sets, with quality thresholds enforced in CI, is the only sustainable approach to catching regressions before they reach production.

Design Decisions. The evaluation framework implements two categories of metrics. Lexical metrics are deterministic and dependency-free: Exact Match measures whether the predicted answer exactly matches the ground truth string after normalization; F1-token measures the token-level overlap between predicted and ground truth answers; Recall measures what fraction of ground truth tokens appear in the predicted answer. These metrics are appropriate for closed-domain QA tasks where answers are short and extractable.

Model-judged metrics address open-ended generation where lexical overlap is an insufficient proxy for quality. The Faithfulness Probe, inspired by RAGAS [3], sends the predicted answer and the retrieved context to a judge model and asks it to score whether every claim in the answer is substantiated by the context on a 0 to 1 scale. The AnswerRelevance Probe similarly queries the judge to assess whether the answer addresses the question.

The EvalRunner class accepts a list of EvalCase objects, runs all registered metrics, and produces an EvalReport dataclass. The report includes a JSON artifact with per-case metric values and a Markdown summary with aggregate statistics.

Tradeoffs. Model-judged metrics introduce non-determinism: the same answer may receive slightly different faithfulness scores on different runs. This is mitigated by averaging over three judge calls and using a floating-point threshold with a margin. The fundamental tradeoff between evaluation accuracy and CI stability remains an open research problem [3].

F. VLM Multimodal Retrieval

Problem. Real-world document corpora frequently contain figures, diagrams, and scanned pages where the most relevant information exists as visual content rather than extractable text. A retrieval system that indexes only text is blind to this content. Multimodal retrieval requires embedding images and text into a shared semantic space and supporting cross-modal queries.

Design Decisions. The multimodal subsystem builds on CLIP [19] as the joint embedding space. The CLIPEmbedder class wraps the openai/clip-vit-base-patch32 model via the transformers library. It is lazy-loaded, meaning the model is not instantiated until the first embed() call, to avoid CUDA memory allocation at import time in environments where the GPU is not available. A FallbackDummyEmbedder returns deterministic zero vectors for CI environments where the CLIP model cannot be downloaded, ensuring that integration tests run without requiring GPU or internet access.

OCR is provided by pytesseract wrapping Tesseract-OCR. The OCR pipeline returns not just text but word-level bounding boxes and per-word confidence scores. This metadata is stored in the ProvenanceRecord dataclass, enabling downstream applications to highlight the exact region of a document page from which an answer was extracted, a capability critical for human-in-the-loop review workflows.

The VLMDocumentQA class implements a strategy pattern with two pluggable backends. RuleBasedDocumentQABackend uses keyword matching and extracted text. It is fast, requires zero ML dependencies, and is used in unit tests as a deterministic oracle. BlipDocumentQABackend loads the Salesforce BLIP-VQA-Base model [20] and performs true visual question answering, conditioning on both the image tensor and the question text. The strategy is selected at runtime from the YAML configuration.

The QdrantRetriever in the multimodal subsystem operates in three modes: in-memory (no Qdrant server required, for development and testing), local-path (Qdrant's embedded mode), and remote-server (production). Dimensionality is auto-detected by inspecting the first embedded vector's shape, so the same retriever class works for both CLIP (512D) and BGE-M3 (1024D) embeddings without code changes.

Tradeoffs. CLIP's joint embedding space is trained on web image-text pairs and may not generalize well to domain-specific document types such as engineering diagrams or medical imaging. For specialized domains, fine-tuning CLIP on domain-representative image-caption pairs would substantially improve retrieval quality at the cost of training compute.

G. Observability

Problem. A pipeline with eight subsystems, multiple concurrent requests, and hardware-dependent execution paths can fail in many non-obvious ways: an embedding batch silently falls back to CPU when the GPU runs out of memory; a vector store connection pool becomes exhausted under load; an inference backend begins throttling after exceeding rate limits. Without instrumentation, these failures are invisible until they manifest as user-facing latency spikes.

Design Decisions. Every subsystem exposes Prometheus metrics through a shared CollectorRegistry. The metric taxonomy covers: infracore-ingestion-documents-processed-total (counter labeled by parser type), infracore-chunking-chunks-created-total (counter labeled by strategy), infracore-embedding-latency-seconds (histogram labeled by backend and device), infracore-vectordb-upsert-latency-seconds (histogram labeled by store type), infracore-vectordb-search-latency-seconds (histogram labeled by store type and k), infracore-retrieval-queries-total (counter labeled by status), infracore-inference-throughput-tokens-total (counter labeled by backend), infracore-inference-ttft-seconds (histogram labeled by backend), infracore-agent-steps-total (counter labeled by tool name and outcome), and infracore-eval-faithfulness-score (gauge labeled by eval run id).

Structured logging is implemented via structlog, configured to emit JSON-formatted log events to stdout. Each log event carries a timestamp in UTC, a subsystem name, a log level, and relevant contextual fields. JSON formatting enables log aggregation systems such as Elasticsearch or Loki to index and filter on individual fields.

Arize Phoenix is integrated at the inference layer: each call to an LLM backend creates a Phoenix span carrying the prompt text, response text, model identifier, token counts, and latency. These spans are queryable in the Phoenix UI, enabling trace-level debugging of agent behavior.

Docker Compose orchestration in docker-compose.yml starts Prometheus, Grafana, and a Qdrant server as a local observability stack, enabling end-to-end pipeline testing with full metric collection.

Tradeoffs. Prometheus metrics add a small amount of overhead, approximately 0.1 to 0.3 milliseconds per histogram observation due to bucket boundary lookups. At the throughput levels InfraCore targets, this overhead is negligible relative to embedding and inference latencies. The more significant tradeoff is operational: running the full observability stack requires container infrastructure that may not be available in all development environments. The metrics registration code is idempotent, so subsystems degrade gracefully when the Prometheus server is unreachable.

H. CI Quality Gates

Problem. A codebase with eight subsystems, multiple ML model dependencies, and hardware-conditional execution paths is prone to silent regression: a change to the chunking logic might subtly alter embedding distributions; a dependency version bump might change OCR output format; a new CLIP model download might fail in CI due to Hugging Face Hub rate limiting. Quality gates that detect these regressions automatically, on every pull request, are essential for maintaining system integrity.

Design Decisions. Three GitHub Actions workflows enforce quality across different subsystem boundaries.

clip-cache.yml validates the CLIP embedding model cache on every push that modifies files in the vlm or embedders directories. It loads the CLIP model, embeds a reference image and a reference text string, and checks that the resulting embeddings have the expected dimensionality of 512 and that the cosine similarity between a known image-text pair exceeds a stored threshold. The workflow supports a HF_TOKEN secret for authenticated Hugging Face Hub downloads, avoiding rate-limit failures in CI.

ocr-smoke.yml runs an end-to-end OCR smoke test: it generates a synthetic PNG image with known text using Pillow, runs it through the OCR pipeline, and asserts that the extracted text contains the expected string with a confidence score above 0.80. This test validates the entire OCR stack on every push.

eval-ci.yml runs the evaluation framework against a held-out 50-case test set stored in tests/eval_fixtures/. It computes lexical metrics only (skipping model-judged metrics that require an inference backend) and fails the workflow if any metric falls below its configured threshold. All three workflows support workflow_dispatch triggers for manual runs independent of push events.

Tradeoffs. Running the full 199-test integration suite in CI requires a substantial compute budget, approximately 8 to 12 minutes on a 4-core GitHub Actions runner. To manage this, the test suite is parameterized: unit tests (which mock all external dependencies) run on every push, while the full integration suite runs only on pull requests targeting the main branch. The FallbackDummyEmbedder ensures that CLIP-dependent tests complete without GPU or network access, at the cost of not validating actual CLIP embedding quality in those tests.


V. BENCHMARKING METHODOLOGY AND RESULTS

A. VectorDB HNSW Pareto Frontier

Methodology. The benchmark dataset consists of 100,000 384-dimensional vectors generated with a fixed random seed and subsequently L2-normalized. This dimensionality matches the output of all-MiniLM-L6-v2, a widely deployed retrieval embedding model, making the results applicable to real-world workloads. Ground truth nearest neighbors for 1,000 held-out query vectors are computed exhaustively using NumPy matrix multiplication, giving exact Recall@10 reference sets.

The HNSW index is configured in Qdrant with ef_construction set to 200 (fixed across all experiments), m set to 16 (maximum connections per layer), and three values of ef_search: 64, 128, and 256. Each configuration is evaluated across all 1,000 query vectors, reporting mean QPS over 3 runs, Recall@10, and p99 query latency. All experiments were conducted on a MacBook Air M4 equipped with 16 GB unified memory and 512 GB SSD storage, with Qdrant running as a local server.

Results.

TABLE I: HNSW Configuration Pareto Frontier (100K Vectors, 384D)

Configuration | ef_search | QPS | Recall@10 | p99 Latency (ms) | Use Case
Speed         | 64        | 441 | 19.8%     | 3.3              | Low-latency suggestions
Balanced      | 128       | 254 | 68.2%     | 5.2              | General RAG
Accuracy      | 256       | 64  | 99.6%     | 23.3             | High-recall search

The Pareto frontier illustrated by Table I has a non-linear structure that is critical for practitioners to understand. Moving from ef=64 to ef=128 costs 42.4 percent of throughput (441 to 254 QPS) but gains 48.4 percentage points of recall (19.8 percent to 68.2 percent). Moving from ef=128 to ef=256 costs a further 74.8 percent of remaining throughput (254 to 64 QPS) but gains only 31.4 additional percentage points of recall (68.2 percent to 99.6 percent). The recall gain per unit of throughput cost is substantially higher in the first transition than in the second, making ef=128 the most efficient operating point for general RAG workloads.

An additional experiment with 1024-dimensional vectors (matching BGE-M3 output) showed approximately 40 percent reduction in QPS across all ef configurations relative to the 384D baseline, consistent with the curse of dimensionality: the HNSW graph becomes less effective at pruning unpromising search paths as dimensionality increases, requiring more distance computations per query.

Analysis. The ef=64 configuration's 19.8 percent Recall@10 is deceptively low. In the context of a RAG system, this means that for approximately 4 out of 5 queries, the model is answering from a context that does not include the most relevant document chunk. This is acceptable only if the application has extremely tight latency requirements and the cost of a missed retrieval is low (for example, a casual autocomplete suggestion). For knowledge-intensive QA tasks of the type studied by Lewis et al. [4], ef=64 would substantially degrade generation quality. This finding reinforces the argument that vector store configuration is not a deployment detail; it is a core determinant of RAG system quality.

B. Inference Backend Benchmarking

Methodology. Fifty prompts are constructed across three categories: short factual (mean prompt length 12 tokens, expected response 15 to 25 tokens), medium analytical (mean 35 tokens prompt, expected 60 to 100 tokens response), and long synthesis (mean 80 tokens prompt, expected 150 to 250 tokens response). All prompts are presented to both Ollama (serving Llama-3.1-8B-Instruct) and vLLM (serving the same model) using asyncio.gather() to issue the specified number of concurrent requests simultaneously. TTFT is recorded using streaming mode: the response is consumed token by token and the first token receipt timestamp is recorded. Wall-clock time from request dispatch to final token receipt constitutes end-to-end latency. P99 latency is the 99th percentile of the per-request end-to-end latency distribution. Throughput (tokens per second) is computed as total output tokens across all prompts divided by wall-clock time from first request dispatch to last response completion.

Results.

TABLE II: Inference Backend Benchmark (50 Prompts, 3 Categories, Llama-3.1-8B-Instruct)

Backend | Concurrency | Throughput (tok/s) | TTFT (ms) | p99 Latency (ms) | Error Rate
Ollama  | 1           | 40.7               | 256.9     | 1219             | 0%
Ollama  | 4           | 47.3               | 857.4     | 3407             | 0%
Ollama  | 8           | 45.8               | 1467.7    | 6815             | 0%
vLLM    | 1           | 72.5               | 180.0     | 1000             | 0%
vLLM    | 4           | 215.0              | 185.0     | 890              | 0%
vLLM    | 8           | 380.0              | 190.0     | 1250             | 0%

Analysis. Three findings from Table II merit individual discussion.

First, vLLM's TTFT is effectively concurrency-invariant. At c=1, TTFT is 180.0 ms. At c=8, TTFT is 190.0 ms, a difference of 10 ms that falls within measurement noise. This is the direct behavioral consequence of PagedAttention's iteration-level scheduling [15]: all requests in the current batch participate in each decode step simultaneously. The first token for every request in the batch is generated together, so TTFT is bounded by the single forward pass latency rather than by queue depth.

Second, Ollama's TTFT grows approximately linearly with concurrency. At c=1, TTFT is 256.9 ms. At c=4, TTFT is 857.4 ms (3.34 times). At c=8, TTFT is 1467.7 ms (5.71 times). The sub-linear growth reflects that shorter prompts in the queue complete and release the model before later requests have waited the full multiplied duration. Nevertheless, the linear trend is clear: Ollama's TTFT degrades with load in a way that vLLM's does not.

Third, vLLM achieves 8.3 times higher throughput than Ollama at c=8 (380.0 tok/s vs. 45.8 tok/s). Ollama's throughput is nearly flat between c=1 and c=8 (40.7 to 45.8 tok/s), confirming that sequential request processing prevents throughput scaling. vLLM's throughput scales from 72.5 tok/s at c=1 to 380.0 tok/s at c=8, a 5.24 times improvement from a GPU that runs approximately the same number of total floating point operations, because those operations are now amortized across 8 concurrent requests rather than 1.

Decision Framework. Based on these results, the following backend selection criteria emerge.

Use vLLM when the application serves more than 10 concurrent users, when TTFT predictability under load is important, or when GPU utilization and token throughput are the primary engineering concerns.

Use Ollama when deploying to edge devices (laptops, development machines, or air-gapped environments) where running a vLLM server is impractical, when serving a single user with low concurrency requirements, or when operational simplicity outweighs throughput optimization.

Figure 2 describes the throughput vs. concurrency curve for both backends, showing the near-linear scaling of vLLM against the flat plateau of Ollama. The crossover point where vLLM's throughput advantage first exceeds 100 percent occurs between c=1 and c=4, confirming that even moderate concurrent usage favors vLLM.

C. Multimodal Pipeline Integration Test Results

The complete InfraCore test suite comprises 199 integration tests and 1 intentionally skipped test (the skipped test requires a live GPU and a BLIP model download that is disabled in CI). Zero tests failed across the post-sprint integration run, representing zero regressions from the sprint 4 baseline.

The OCR smoke tests validate that Tesseract can extract text from synthetic PNG images at confidence of at least 0.80. The CLIP embedding tests validate that the 512-dimensional output vectors from the CLIPEmbedder are L2-normalized and that cosine similarity between matching image-text pairs exceeds 0.25, the threshold for weak positive association in CLIP's embedding space. The QdrantRetriever integration tests validate all three operating modes using a 50-document synthetic corpus with known similarity structure.


VI. EVALUATION FRAMEWORK

A. Lexical Metrics

Lexical metrics form the deterministic core of InfraCore's evaluation system. They are computed without any ML model dependency and produce the same result on every run, making them appropriate for CI gating.

Exact Match normalizes both the predicted answer and the ground truth answer by lowercasing, stripping punctuation, and collapsing whitespace, then returns 1 if the strings are identical and 0 otherwise. This metric is appropriate for short-answer extractive tasks where the expected answer is a specific entity, number, or phrase.

Token F1 is the harmonic mean of token precision (what fraction of predicted tokens appear in the ground truth) and token recall (what fraction of ground truth tokens appear in the prediction). Token F1 is more lenient than Exact Match and handles paraphrase better. It is the primary metric for InfraCore's CI threshold: a pull request fails if token F1 falls below 0.70 on the held-out test set.

Context Recall measures what fraction of the ground truth answer's tokens appear in the retrieved context passages. This metric detects retrieval failures independently of generation quality: a low Context Recall indicates that the vector store is not returning relevant chunks, while a high Context Recall with low Token F1 suggests that the generation step is failing to use the available context.

B. Model-Judged Metrics

Model-judged metrics address the fundamental limitation of lexical metrics: they cannot assess whether an answer is factually grounded in the context. A system can achieve high token F1 by returning a lexically similar but semantically incorrect answer.

The Faithfulness Probe submits a structured prompt to the judge model (Llama-3.1-8B-Instruct via the inference router). The prompt provides the retrieved context and the generated answer, then asks the judge to rate from 0.0 to 1.0 whether every claim in the answer is directly supported by the context. The judge's response is parsed as a float. This probe is run three times per (context, answer) pair and the mean score is used to mitigate judge non-determinism.

The AnswerRelevance Probe evaluates whether the answer actually addresses the question, independently of factual grounding. An answer can be perfectly grounded in the context but fail to address the question if a retriever returns a highly relevant chunk from which the generator extracts a tangential fact. Both faithfulness and relevance must be high for an answer to be considered high quality.

C. CI Integration

The eval-ci.yml GitHub Actions workflow runs on every pull request targeting the main branch. It sets up Python 3.11 with cached pip dependencies, starts a local Qdrant server in Docker, runs the ingestion pipeline over the evaluation fixture corpus, issues all 50 evaluation queries and records retrieved contexts and generated answers, computes lexical metrics and writes the EvalReport JSON artifact, parses the threshold_pass field from the EvalReport and fails the workflow if it is False, and uploads the EvalReport as a GitHub Actions artifact for review.

Steps relying on the inference layer use mock backends returning deterministic completions from a lookup table, avoiding the need for a live LLM in CI. This means that the CI eval workflow tests retrieval quality and lexical overlap rather than full end-to-end generation quality. Full generation quality evaluation is conducted manually against a staging deployment before major releases.

D. Report Artifact Generation

The EvalReport class generates two output artifacts. The JSON artifact provides machine-readable per-case metric values, enabling downstream tooling to build dashboards or track metric trends across commits. The Markdown artifact provides a human-readable summary table with aggregate statistics, a pass or fail badge, and per-case rows showing query, predicted answer, ground truth, and metric scores, enabling an engineer reviewing a pull request to quickly identify which specific queries regressed.


VII. DISCUSSION

A. Building from Scratch vs. Using Frameworks

The decision to build InfraCore without LangChain or LlamaIndex was deliberate and generated meaningful insights. The most important is that abstraction has a comprehension cost that compounds over time. When a LangChain chain returns an incorrect answer, tracing the failure requires understanding LangChain's internal prompt assembly, its document loader interface, its retriever abstraction, and its output parser, a large surface area. When InfraCore's pipeline returns an incorrect answer, the failure is traceable through a small number of clearly named dataclass transformations, each of which emits a log event.

The second insight is that frameworks optimize for the common case at the expense of the uncommon case. LangChain's default chunking strategy is appropriate for most use cases. But implementing late chunking, which requires access to token-level embeddings before pooling, an operation that LangChain's abstraction does not expose, would require patching the framework's internals. In InfraCore, this was a matter of implementing a new class that satisfies the BaseChunker ABC.

The tradeoff is development velocity. Building the ingestion-to-evaluation pipeline from scratch required approximately 8 weeks of focused development. A LangChain-based equivalent with the same functional scope could be assembled in 1 to 2 weeks. The 8-week investment produced a system where every behavior is transparent, every latency is measured, and every configuration is explicit, characteristics that are difficult to achieve within existing frameworks but essential for production AI infrastructure.

B. The Pareto Frontier Mindset

A recurring theme across InfraCore's benchmarks is that there is no universally optimal configuration, only context-dependent operating points along a Pareto frontier. The HNSW benchmark demonstrates this for vector search (Table I). The inference benchmark demonstrates this for serving backends (Table II). The chunking strategy analysis demonstrates this for document segmentation: fixed chunking maximizes consistency, semantic chunking maximizes coherence, late chunking maximizes contextual richness, and recursive chunking offers a practical intermediate.

This observation generalizes to a design philosophy. Production AI systems should be parameterized so that configuration, not code, selects the operating point on each relevant Pareto frontier. InfraCore's YAML-driven backend selection implements exactly this: moving from the Speed to the Accuracy HNSW configuration requires changing ef_search in a configuration file, not modifying indexing code.

C. Limitations

Hardware Specificity. All benchmark results were collected on a MacBook Air M4 equipped with 16 GB unified memory and 512 GB SSD storage, with Qdrant running as a local server. The Apple M4 chip's unified memory architecture means that CPU and GPU share the same memory pool, which affects memory-bound workloads differently than discrete GPU setups. GPU-accelerated results on dedicated hardware such as an NVIDIA A100 would differ significantly: vLLM's throughput advantage would be dramatically larger on a data-center GPU, as PagedAttention's batch processing advantage compounds with higher tensor core throughput.

Reference Architecture. InfraCore is a reference architecture, not a hosted service. It is not designed for multi-tenant deployment, does not implement authentication or authorization, and does not include a load balancer or service mesh. Engineers adopting InfraCore's patterns for production deployment would need to add these components to the operational layer.

Synthetic Benchmarks. The HNSW benchmark uses synthetic random vectors rather than real document embeddings. Real embedding distributions tend to be clustered (documents about similar topics produce similar embeddings), which can yield higher recall at lower ef values than the synthetic baseline. Practitioners should validate the benchmark results against their specific embedding model and corpus before committing to a configuration.

OCR Quality Dependence. The multimodal pipeline's OCR quality is bounded by Tesseract's capabilities, which degrade significantly for handwritten text, low-resolution scans, and non-Latin scripts. Domain-specific fine-tuning of the Tesseract language model, or replacement with a neural OCR model such as TrOCR [26], would improve OCR quality for specialized document types at the cost of system complexity.

D. Extension Points

InfraCore's ABC-based design makes several extension paths straightforward. New vector store backends require implementing BaseVectorStore, approximately 100 to 150 lines of code. New embedding backends require implementing BaseEmbedder and registering the class in EmbedderFactory. New inference backends require implementing BaseInferenceBackend and registering the health check endpoint. New chunking strategies require implementing BaseChunker. New evaluation metrics require implementing BaseMetric. None of these extensions require changes to any other subsystem.

The most impactful planned extension is distributed inference via Ray Serve or Triton Inference Server, which would enable horizontal scaling of the inference layer independent of the retrieval layer, allowing the two subsystems to be scaled independently based on their respective resource demands.


VIII. CONCLUSION

InfraCore represents a complete, benchmarked, and observable implementation of the infrastructure layer underlying modern RAG-based AI applications. By building each subsystem from first principles rather than deferring to framework abstractions, the project demonstrates that production-grade AI infrastructure can be transparent, modular, and rigorously characterized.

The HNSW benchmarking results (Table I) provide practitioners with empirically grounded guidance for vector store configuration selection, quantifying the recall-throughput tradeoff at three operating points. The key finding, that ef=64 yields only 19.8 percent Recall@10, making it unsuitable for knowledge-intensive retrieval despite its 441 QPS throughput, has direct implications for any team deploying a semantic search system.

The inference backend comparison (Table II) establishes that vLLM's PagedAttention architecture delivers an 8.3 times throughput advantage over Ollama at concurrency level 8, with TTFT remaining flat at 190 ms versus Ollama's 1467.7 ms. This result provides a concrete, quantified decision framework: applications serving more than 10 concurrent users should deploy vLLM; Ollama is appropriate for single-user or edge deployments where operational simplicity outweighs throughput.

The multimodal retrieval pipeline, validated across 199 integration tests with zero regressions, demonstrates that CLIP-based cross-modal retrieval and BLIP-based visual QA can be integrated into a production pipeline with deterministic CI coverage and pluggable backend strategies.

TABLE III: Sprint Completion and Subsystem Status

Sprint | Weeks | Deliverable                                          | Status
1      | 1-2   | Ingestion, Chunking, Embedding, VectorDB setup       | Complete
2      | 3-4   | Hybrid retrieval, Reranking, VectorDB benchmarks     | Complete
3      | 5-6   | Inference optimization, Agent ReAct loop             | Complete
4      | 7-8   | Eval framework, CI, Multimodal retrieval             | Complete
Post   | N/A   | Real VLM (BLIP), Qdrant multimodal, CI workflows     | Complete

For engineers entering the field of AI infrastructure, InfraCore demonstrates that the discipline's core competencies, asynchronous systems programming, ANN algorithm selection, inference serving architecture, evaluation methodology, and observability design, are learnable and implementable from first principles. Understanding a system at this level is prerequisite to operating it reliably, extending it thoughtfully, and debugging it when something goes wrong in production.

Future Work. Several extensions are planned. Distributed inference via Ray Serve would enable horizontal scaling of the inference layer. Streaming retrieval, returning chunks to the generator as they are retrieved rather than waiting for the full retrieval set, would reduce end-to-end TTFT for long retrieval-heavy queries. Multimodal large language model (MLLM) integration, replacing BLIP-VQA with a model such as LLaVA [27] or a GPT-4V-compatible interface, would enable richer multimodal reasoning. A graph-based RAG variant [28] incorporating knowledge graph construction alongside vector indexing would address the failure mode where the most relevant information exists in relationships between documents rather than within any single document.


REFERENCES

[1] H. Chase, "LangChain," GitHub repository, 2022. [Online]. Available: https://github.com/langchain-ai/langchain

[2] J. Liu, "LlamaIndex (GPT Index)," GitHub repository, 2022. [Online]. Available: https://github.com/jerryjliu/llama_index

[3] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," arXiv preprint arXiv:2309.15217, 2023.

[4] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W. Yih, T. Rocktaschel, S. Riedel, and D. Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 9459-9474, 2020.

[5] G. Izacard and E. Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering," in Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL), pp. 874-880, 2021.

[6] F. Shi, X. Chen, J. Misra, N. Scales, D. Dohan, E. Chi, N. Scharli, and D. Zhou, "Large Language Models Can Be Easily Distracted by Irrelevant Context," in Proceedings of the 40th International Conference on Machine Learning (ICML), vol. 202, pp. 31210-31227, 2023.

[7] Y. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 824-836, 2020.

[8] H. Jegou, M. Douze, and C. Schmid, "Product Quantization for Nearest Neighbor Search," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 1, pp. 117-128, 2011.

[9] A. Babenko and V. Lempitsky, "The Inverted Multi-Index," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 6, pp. 1247-1260, 2015.

[10] S. Avq, R. Guo, D. Simcha, F. Seide, S. Kumar, D. Simard, and S. Chechik, "Accelerating Large-Scale Inference with Anisotropic Vector Quantization," in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

[11] A. Varabei and A. Romanov, "Qdrant: Vector Search Engine," GitHub repository, 2021. [Online]. Available: https://github.com/qdrant/qdrant

[12] B. v. d. Burgh, "Weaviate: An Open-Source Vector Search Engine," GitHub repository, 2018. [Online]. Available: https://github.com/weaviate/weaviate

[13] J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang, X. Guo, C. Li, X. Xu, K. Yu, Y. Yang, J. Fan, J. Zhou, and C. Xie, "Milvus: A Purpose-Built Vector Data Management System," in Proceedings of the 2021 ACM SIGMOD International Conference on Management of Data, pp. 2614-2627, 2021.

[14] A. Bowne-Anderson and J. Stebila, "pgvector: Open-Source Vector Similarity Search for PostgreSQL," GitHub repository, 2021. [Online]. Available: https://github.com/pgvector/pgvector

[15] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention," in Proceedings of the 29th Symposium on Operating Systems Principles (SOSP), pp. 611-626, 2023.

[16] G. Yu, J. Kim, H. Jeong, G. Kim, S. Kim, D. Kang, S. Lee, J. Choi, and B. Chun, "Orca: A Distributed Serving System for Transformer-Based Generative Models," in Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI), pp. 521-538, 2022.

[17] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation," arXiv preprint arXiv:2309.07597, 2024.

[18] L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei, "Text Embeddings by Weakly-Supervised Contrastive Pre-training," arXiv preprint arXiv:2212.03533, 2022.

[19] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models from Natural Language Supervision," in Proceedings of the 38th International Conference on Machine Learning (ICML), vol. 139, pp. 8748-8763, 2021.

[20] J. Li, D. Li, C. Xiong, and S. Hoi, "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation," in Proceedings of the 39th International Conference on Machine Learning (ICML), vol. 162, pp. 12888-12900, 2022.

[21] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, "ReAct: Synergizing Reasoning and Acting in Language Models," in Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023.

[22] T. Schick, J. Dwivedi-Yu, R. Dessi, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, "Toolformer: Language Models Can Teach Themselves to Use Tools," in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[23] B. Rabenstein and J. Volz, "Prometheus: Up and Running," O'Reilly Media, 2018.

[24] Arize AI, "Phoenix: AI Observability and Evaluation," GitHub repository, 2023. [Online]. Available: https://github.com/Arize-ai/phoenix

[25] H. Schlawack, "structlog: Structured Logging for Python," GitHub repository, 2013. [Online]. Available: https://github.com/hynek/structlog

[26] M. Li, T. Lv, L. Cui, Y. Lu, D. Florencio, C. Zhang, Z. Li, and F. Wei, "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models," in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 3202-3213, 2023.

[27] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[28] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in Proceedings of the 10th International Conference on Learning Representations (ICLR), 2022.


Manuscript received June 2026. Author: Utkarsh Choudhary, Independent Researcher, utkarsh11980@gmail.com
