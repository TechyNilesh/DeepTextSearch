"""PostgreSQL + pgvector based vector store."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class PostgresStore(BaseVectorStore):
    """PostgreSQL vector store using pgvector extension.

    Requires: pip install psycopg2-binary pgvector

    Stores vectors alongside metadata in a JSONB column. You can also define
    additional dedicated columns for structured metadata fields using
    `metadata_schema`, which enables direct SQL indexing and filtering.

    Args:
        connection_string: PostgreSQL connection string.
        table_name: Name of the vector table.
        dimension: Vector dimension.
        metadata_schema: Optional dict mapping column names to SQL types for
            dedicated metadata columns. These columns are created alongside
            the default JSONB `metadata` column for fast filtering.

    Example:
        >>> # Basic — metadata stored as JSONB
        >>> store = PostgresStore(
        ...     connection_string="postgresql://user:pass@localhost/mydb",
        ...     table_name="article_vectors",
        ... )

        >>> # With dedicated metadata columns for fast filtering
        >>> store = PostgresStore(
        ...     connection_string="postgresql://user:pass@localhost/mydb",
        ...     table_name="article_vectors",
        ...     metadata_schema={
        ...         "category": "TEXT",
        ...         "language": "TEXT",
        ...         "year": "INTEGER",
        ...         "source": "TEXT",
        ...     },
        ... )
    """

    def __init__(
        self,
        connection_string: str = "postgresql://localhost:5432/deeptextsearch",
        table_name: str = "text_vectors",
        dimension: int = 1024,
        metadata_schema: Optional[Dict[str, str]] = None,
    ):
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError:
            raise ImportError(
                "psycopg2-binary and pgvector are required for PostgresStore.\n"
                "Install with: pip install 'DeepTextSearch[postgres]'"
            )

        self.connection_string = connection_string
        self.table_name = table_name
        self.dimension = dimension
        self.metadata_schema = metadata_schema or {}

        self.conn = psycopg2.connect(connection_string)
        self.conn.autocommit = True

        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        register_vector(self.conn)
        self._create_table()

        logger.info(f"PostgreSQL store '{table_name}' ready ({self.count()} vectors)")

    def _create_table(self) -> None:
        """Create the vector table with optional dedicated metadata columns."""
        schema_columns = ""
        if self.metadata_schema:
            cols = [f"{col} {dtype}" for col, dtype in self.metadata_schema.items()]
            schema_columns = ", " + ", ".join(cols)

        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    text_content TEXT{schema_columns}
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            # Create indexes on dedicated metadata columns
            for col in self.metadata_schema:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_{col}_idx
                    ON {self.table_name} ({col})
                """)

    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        vectors = vectors.astype(np.float32)

        # Build column lists for dedicated schema columns
        schema_cols = list(self.metadata_schema.keys())
        extra_col_names = (", " + ", ".join(schema_cols)) if schema_cols else ""
        extra_placeholders = (", " + ", ".join(["%s"] * len(schema_cols))) if schema_cols else ""
        extra_update = ""
        if schema_cols:
            extra_update = ", " + ", ".join(
                [f"{col} = EXCLUDED.{col}" for col in schema_cols]
            )

        with self.conn.cursor() as cur:
            for i, (id_, vec) in enumerate(zip(ids, vectors)):
                meta = metadata[i] if metadata else {}
                text_content = meta.pop("_text", None)

                # Extract values for dedicated columns
                extra_values = [meta.get(col) for col in schema_cols]

                cur.execute(
                    f"""
                    INSERT INTO {self.table_name}
                        (id, embedding, metadata, text_content{extra_col_names})
                    VALUES (%s, %s, %s, %s{extra_placeholders})
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        text_content = EXCLUDED.text_content{extra_update}
                    """,
                    [id_, vec.tolist(), json.dumps(meta), text_content] + extra_values,
                )
        logger.info(f"Added {len(ids)} vectors to PostgreSQL")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query = query_vector.astype(np.float32).tolist()

        where_parts = []
        filter_values = []
        if filters:
            for key, value in filters.items():
                # Use dedicated column if it exists, otherwise filter via JSONB
                if key in self.metadata_schema:
                    where_parts.append(f"{key} = %s")
                else:
                    where_parts.append(f"metadata->>'{key}' = %s")
                filter_values.append(value)

        where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, 1 - (embedding <=> %s::vector) AS score, metadata
                FROM {self.table_name}
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [query] + filter_values + [query, k],
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            meta = row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {}
            results.append({
                "id": row[0],
                "score": float(row[1]),
                "metadata": meta,
            })
        return results

    def delete(self, ids: List[str]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
                (ids,),
            )

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return cur.fetchone()[0]

    def save(self, path: str) -> None:
        logger.info("PostgreSQL persists automatically (no manual save needed)")

    def load(self, path: str) -> None:
        logger.info("PostgreSQL loads automatically from the database connection")
