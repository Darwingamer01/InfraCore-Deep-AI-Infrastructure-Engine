"""
INFRACORE — PgVectorStore

PostgreSQL + pgvector implementation using asyncpg.
Alternative to Qdrant for users preferring SQL databases.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

import asyncpg
import numpy as np
from prometheus_client import Counter, Histogram
from pydantic import ConfigDict, Field

from src.infracore.vectordb.base import BaseVectorStore, SearchResult, VectorStoreConfig

# Prometheus metrics
pg_operations = Counter(
    "pgvector_operations_total",
    "Total pgvector operations",
    ["operation", "table"],
)
pg_latency = Histogram(
    "pgvector_latency_seconds",
    "PgVector operation latency",
    ["operation", "table"],
)


class PgVectorConfig(VectorStoreConfig):
    """PostgreSQL + pgvector configuration."""

    model_config = ConfigDict(frozen=True)

    store_type: str = Field(default="pgvector", description="Store type")
    dsn: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/infracore",
        description="PostgreSQL connection string",
    )
    table_name: str = Field(default="embeddings", description="Table name")
    index_type: str = Field(default="hnsw", description="Index type: hnsw or ivfflat")
    hnsw_m: int = Field(default=16, description="HNSW M parameter")
    hnsw_ef_construct: int = Field(default=64, description="HNSW ef_construct")
    ivfflat_lists: int = Field(default=100, description="IVFFlat lists parameter")


class PgVectorStore(BaseVectorStore):
    """
    PostgreSQL vector store using pgvector extension and asyncpg.

    Config:
        dsn: PostgreSQL connection string
        table_name: Table name for embeddings
        vector_size: Embedding dimension
        index_type: "hnsw" or "ivfflat"
        hnsw_m, hnsw_ef_construct, ivfflat_lists: Index parameters

    Example:
        cfg = PgVectorConfig(table_name="documents", vector_size=1024)
        store = PgVectorStore(cfg)
        await store.create_table()
        await store.upsert(vectors, payloads)
        results = await store.search(query_vector, top_k=5)
    """

    def __init__(self, config: PgVectorConfig):
        super().__init__(config)
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Lazy-initialize connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.config.dsn, min_size=5, max_size=20)
        return self.pool

    async def create_table(self) -> None:
        """Create embeddings table with pgvector column and index."""
        import time

        start = time.perf_counter()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.config.vector_size}) NOT NULL,
                    payload JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """
            await conn.execute(create_table_sql)

            # Create index
            index_name = f"{self.config.table_name}_embedding_idx"
            if self.config.index_type == "hnsw":
                create_index_sql = f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.config.table_name}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construct})
                """
            else:  # ivfflat
                create_index_sql = f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.config.table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.ivfflat_lists})
                """

            await conn.execute(create_index_sql)

        elapsed = time.perf_counter() - start
        pg_operations.labels(operation="create_table", table=self.config.table_name).inc()
        pg_latency.labels(operation="create_table", table=self.config.table_name).observe(elapsed)

    async def upsert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Upsert vectors and payloads using INSERT ... ON CONFLICT.

        Args:
            vectors: shape (N, vector_size)
            payloads: metadata for each vector
            ids: optional IDs; auto-generated if None

        Returns:
            Count of upserted vectors
        """
        import time

        start = time.perf_counter()

        N = len(vectors)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(N)]

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Prepare insert statement
            insert_sql = f"""
                INSERT INTO {self.config.table_name} (id, embedding, payload)
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload
            """

            # Build batch of tuples: (id, embedding_list, payload_json)
            records = [
                (ids[i], vectors[i].tolist(), json.dumps(payloads[i]))
                for i in range(N)
            ]

            # Batch insert using executemany
            await conn.executemany(insert_sql, records)

        elapsed = time.perf_counter() - start
        pg_operations.labels(operation="upsert", table=self.config.table_name).inc()
        pg_latency.labels(operation="upsert", table=self.config.table_name).observe(elapsed)

        return N

    async def search(
        self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors using pgvector cosine distance.

        Args:
            query_vector: shape (vector_size,)
            top_k: number of results
            filter: optional WHERE clause dict (not used, for compatibility)

        Returns:
            List[SearchResult] sorted by score (best first)
        """
        import time

        start = time.perf_counter()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Query using <-> (cosine distance operator)
            search_sql = f"""
                SELECT id, embedding <-> $1::vector AS distance, payload
                FROM {self.config.table_name}
                ORDER BY distance
                LIMIT $2
            """

            rows = await conn.fetch(search_sql, query_vector.tolist(), top_k)

        # Convert to SearchResult (score = 1 - distance for similarity)
        results = [
            SearchResult(
                id=row["id"],
                score=1.0 - row["distance"],  # Convert distance to similarity
                payload=json.loads(row["payload"]) if row["payload"] else {},
            )
            for row in rows
        ]

        elapsed = time.perf_counter() - start
        pg_operations.labels(operation="search", table=self.config.table_name).inc()
        pg_latency.labels(operation="search", table=self.config.table_name).observe(elapsed)

        return results

    async def delete(self, ids: List[str]) -> int:
        """Delete vectors by ID."""
        import time

        start = time.perf_counter()

        if not ids:
            return 0

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Use DELETE with IN clause
            placeholders = ", ".join([f"${i+1}" for i in range(len(ids))])
            delete_sql = f"DELETE FROM {self.config.table_name} WHERE id IN ({placeholders})"
            result = await conn.execute(delete_sql, *ids)

        # Parse result string like "DELETE 5"
        deleted_count = len(ids)

        elapsed = time.perf_counter() - start
        pg_operations.labels(operation="delete", table=self.config.table_name).inc()
        pg_latency.labels(operation="delete", table=self.config.table_name).observe(elapsed)

        return deleted_count

    async def count(self) -> int:
        """Return total vector count."""
        import time

        start = time.perf_counter()

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {self.config.table_name}")

        count = result if result else 0

        elapsed = time.perf_counter() - start
        pg_operations.labels(operation="count", table=self.config.table_name).inc()
        pg_latency.labels(operation="count", table=self.config.table_name).observe(elapsed)

        return count

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
