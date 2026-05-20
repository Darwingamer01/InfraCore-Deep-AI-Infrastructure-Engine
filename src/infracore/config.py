"""
INFRACORE — Compatibility Config Re-Exports.

Provides legacy imports used by integration and smoke tests.
"""

from src.infracore.embedding.base import EmbedConfig
from src.infracore.vectordb.qdrant_store import QdrantConfig
from src.infracore.vectordb.pgvector_store import PgVectorConfig

# Backwards-compatible aliases for older call sites.
BGEConfig = EmbedConfig
