Multimodal Subsystem — Release Notes (2026-05-23)

Summary
-------
This release promotes the multimodal document intelligence subsystem to a production-ready milestone with persistent retrieval via Qdrant, optional model-backed VQA via BLIP, robust OCR provenance, and CI hardening for large model downloads.

Highlights
----------
- CLIP embeddings: configurable, device-aware image/text embeddings
- OCR pipeline: pytesseract integration with word-level metadata and graceful fallbacks
- VLMDocumentQA: preserved rule-based deterministic backend plus optional BLIP backend (`Salesforce/blip-vqa-base`) validated locally
- Provenance: full source metadata (source_type, doc_id, page, bounding_box, coordinates, confidence)
- CI hardening: `HF_TOKEN` secret support added to workflows to avoid unauthenticated HF downloads
- Qdrant persistence: new `QdrantRetriever` supporting in-memory, local-path, and remote server modes; payload-rich points for provenance

Files Added / Updated
---------------------
- Added: `src/infracore/multimodal/qdrant_retriever.py` — Qdrant-backed retriever
- Updated: `src/infracore/multimodal/__init__.py` — export QdrantRetriever
- Added tests: `tests/unit/test_qdrant_retriever.py` (in-memory), `tests/integration/test_qdrant_server.py` (optional server)
- Updated: `.github/workflows/clip-cache.yml` and `.github/workflows/ocr-smoke.yml` to accept `HF_TOKEN` secret
- Updated: `.env.example` to document `HF_TOKEN` for local development
- Updated: `MULTIMODAL_MILESTONE.md` with BLIP validation and Qdrant persistence notes

Operational notes
-----------------
- BLIP VQA validation completed locally (model ~1.5GB). Set `HF_TOKEN` as a repo secret and local env variable to avoid HF rate limits.
- When pointing to an existing remote Qdrant server, ensure the collection `vectors` size matches your chosen embedder (CLIP is commonly 512-dim). Server-mode tests may fail if collection dimensions differ; recreate the collection with the correct `VectorParams` if needed.

How to verify locally
---------------------
```bash
# Set HF token for model downloads
export HF_TOKEN=hf_...

# Run unit tests
source .venv/bin/activate
PYTHONPATH=src pytest tests/unit/test_qdrant_retriever.py -q

# Optional: Run BLIP integration test (requires transformers + torch)
PYTHONPATH=src pytest tests/integration/test_multimodal_pipeline.py::test_blip_backend_optional -q -s -v

# Optional: Run Qdrant server-mode integration tests (requires local Qdrant at :6333)
# docker run -p 6333:6333 qdrant/qdrant
PYTHONPATH=src pytest tests/integration/test_qdrant_server.py -q -vv -s
```

Acknowledgements
----------------
This milestone ties together the rule-based QA, model-backed VQA, OCR provenance, embedding pipeline, and persistent retrieval into a coherent, test-covered platform component.
