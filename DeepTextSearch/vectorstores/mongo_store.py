"""MongoDB Atlas Vector Search based vector store."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class MongoStore(BaseVectorStore):
    """MongoDB vector store using Atlas Vector Search.

    Requires: pip install pymongo

    Stores vectors and metadata in MongoDB documents. Metadata fields
    are stored as top-level document fields for native MongoDB querying
    and indexing.

    Args:
        connection_string: MongoDB connection string.
        database_name: Database name.
        collection_name: Collection name.
        index_name: Atlas Vector Search index name.
        dimension: Vector dimension.
        metadata_fields: Optional list of metadata field names to store
            as top-level document fields (for MongoDB indexing/querying).
            If not specified, all metadata is stored in a nested `metadata` dict.

    Example:
        >>> # Basic
        >>> store = MongoStore(
        ...     connection_string="mongodb+srv://user:pass@cluster.mongodb.net",
        ...     database_name="search_db",
        ...     collection_name="articles",
        ... )

        >>> # With top-level metadata fields for fast querying
        >>> store = MongoStore(
        ...     connection_string="mongodb+srv://user:pass@cluster.mongodb.net",
        ...     database_name="search_db",
        ...     collection_name="articles",
        ...     metadata_fields=["category", "language", "author", "year"],
        ... )
    """

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017",
        database_name: str = "deeptextsearch",
        collection_name: str = "text_vectors",
        index_name: str = "vector_index",
        dimension: int = 1024,
        metadata_fields: Optional[List[str]] = None,
    ):
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoStore.\n"
                "Install with: pip install 'DeepTextSearch[mongo]'"
            )

        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.index_name = index_name
        self.dimension = dimension
        self.metadata_fields = metadata_fields or []

        # Create indexes on doc_id and metadata fields
        self.collection.create_index("doc_id", unique=True, sparse=True)
        for field in self.metadata_fields:
            self.collection.create_index(field)

        logger.info(
            f"MongoDB store '{database_name}.{collection_name}' ready ({self.count()} vectors)"
        )

    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        from pymongo import UpdateOne

        vectors = vectors.astype(np.float32)
        operations = []

        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            meta = metadata[i] if metadata else {}

            doc = {
                "doc_id": id_,
                "embedding": vec.tolist(),
            }

            # Store specified fields as top-level for fast querying
            remaining_meta = {}
            for key, value in meta.items():
                if key in self.metadata_fields:
                    doc[key] = value
                else:
                    remaining_meta[key] = value

            doc["metadata"] = remaining_meta

            operations.append(
                UpdateOne(
                    {"doc_id": doc["doc_id"]},
                    {"$set": doc},
                    upsert=True,
                )
            )

        if operations:
            self.collection.bulk_write(operations)
        logger.info(f"Added {len(ids)} vectors to MongoDB")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query = query_vector.astype(np.float32).tolist()

        # Build projection to include top-level metadata fields
        project = {
            "doc_id": 1,
            "metadata": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
        for field in self.metadata_fields:
            project[field] = 1

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": query,
                    "numCandidates": k * 10,
                    "limit": k,
                }
            },
            {"$project": project},
        ]

        # Apply filters on top-level metadata fields or nested metadata
        if filters:
            match_conditions = {}
            for key, value in filters.items():
                if key in self.metadata_fields:
                    match_conditions[key] = value
                else:
                    match_conditions[f"metadata.{key}"] = value
            pipeline.insert(1, {"$match": match_conditions})

        results = []
        for doc in self.collection.aggregate(pipeline):
            # Merge top-level metadata fields back into metadata dict
            meta = doc.get("metadata", {})
            for field in self.metadata_fields:
                if field in doc:
                    meta[field] = doc[field]

            results.append({
                "id": doc.get("doc_id", str(doc.get("_id"))),
                "score": float(doc.get("score", 0.0)),
                "metadata": meta,
            })
        return results

    def delete(self, ids: List[str]) -> None:
        self.collection.delete_many({"doc_id": {"$in": ids}})

    def count(self) -> int:
        return self.collection.count_documents({})

    def save(self, path: str) -> None:
        logger.info("MongoDB persists automatically (no manual save needed)")

    def load(self, path: str) -> None:
        logger.info("MongoDB loads automatically from the database connection")
