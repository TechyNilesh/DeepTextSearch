"""Qdrant-based vector store."""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    """Qdrant vector store backend.

    Requires: pip install qdrant-client

    Supports in-memory, local persistent, and remote server modes.

    Args:
        collection_name: Name of the Qdrant collection.
        location: Qdrant server URL or ':memory:' for in-memory. None defaults to ':memory:'.
        path: Path for local persistent storage (alternative to server).
        dimension: Vector dimension.
    """

    def __init__(
        self,
        collection_name: str = "deep_text_search",
        location: Optional[str] = None,
        path: Optional[str] = None,
        dimension: int = 1024,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantStore.\n"
                "Install with: pip install 'DeepTextSearch[qdrant]'"
            )

        if path:
            self.client = QdrantClient(path=path)
        elif location:
            self.client = QdrantClient(url=location)
        else:
            self.client = QdrantClient(location=":memory:")

        self.collection_name = collection_name
        self.dimension = dimension

        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

        logger.info(f"Qdrant collection '{collection_name}' ready ({self.count()} vectors)")

    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        vectors = vectors.astype(np.float32)
        points = []
        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            payload = metadata[i] if metadata else {}
            payload["_original_id"] = id_
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, id_)),
                vector=vec.tolist(),
                payload=payload,
            ))

        batch_size = 500
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i + batch_size],
            )
        logger.info(f"Added {len(ids)} vectors to Qdrant")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filters:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.astype(np.float32).tolist(),
            limit=k,
            query_filter=query_filter,
        )

        output = []
        for hit in results:
            payload = dict(hit.payload) if hit.payload else {}
            original_id = payload.pop("_original_id", str(hit.id))
            output.append({
                "id": original_id,
                "score": hit.score,
                "metadata": payload,
            })
        return output

    def delete(self, ids: List[str]) -> None:
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, id_)) for id_ in ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
        )

    def count(self) -> int:
        return self.client.get_collection(self.collection_name).points_count

    def save(self, path: str) -> None:
        logger.info("Qdrant persists automatically based on client configuration")

    def load(self, path: str) -> None:
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=path)
