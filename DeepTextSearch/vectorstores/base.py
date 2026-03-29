"""Abstract vector store interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseVectorStore(ABC):
    """Abstract interface for vector storage backends.

    All stores must support:
    - Adding vectors with IDs and optional metadata
    - Searching by vector with optional metadata filters
    - Persistence (save/load)
    - Deletion by ID
    """

    @abstractmethod
    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to the store.

        Args:
            ids: Unique identifiers for each vector.
            vectors: (N, D) float32 array of vectors.
            metadata: Optional metadata per vector.
        """

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: 1-D query vector.
            k: Number of results.
            filters: Metadata filters (store-specific).

        Returns:
            List of dicts with 'id', 'score', and 'metadata'.
        """

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the store to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the store from disk."""
