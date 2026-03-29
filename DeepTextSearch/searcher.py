"""Hybrid text search engine combining dense (vector store) and sparse (BM25) retrieval."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from rank_bm25 import BM25Okapi

from .embedder import TextEmbedder


class SearchResult:
    """A single search result.

    Attributes:
        index: Position in the original corpus.
        text: The matched text.
        score: Combined relevance score.
        metadata: Associated metadata dict.
    """

    __slots__ = ("index", "text", "score", "metadata")

    def __init__(self, index: int, text: str, score: float, metadata: dict):
        self.index = index
        self.text = text
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "score": round(float(self.score), 6),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"SearchResult(index={self.index}, score={self.score:.4f}, text='{self.text[:60]}...')"


class TextSearch:
    """Hybrid search engine combining semantic (dense) and keyword (BM25) search.

    Works with any vector store backend (FAISS, ChromaDB, Qdrant, or custom).

    Supports three search modes:
    - "hybrid" (default): Combines dense + BM25 with Reciprocal Rank Fusion.
    - "dense": Pure semantic similarity search via vector store.
    - "bm25": Pure keyword search via BM25.

    Args:
        embedder: A TextEmbedder instance with an indexed corpus.
        mode: Search mode - "hybrid", "dense", or "bm25".
        bm25_weight: Weight for BM25 scores in hybrid fusion (0.0 to 1.0).
        dense_weight: Weight for dense scores in hybrid fusion (0.0 to 1.0).
        rrf_k: Constant for Reciprocal Rank Fusion (default 60).

    Example:
        >>> embedder = TextEmbedder()
        >>> embedder.index(["Python is great", "Java is popular", "AI is the future"])
        >>> search = TextSearch(embedder)
        >>> results = search.search("programming language", top_n=2)
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        mode: str = "hybrid",
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        rrf_k: int = 60,
    ):
        if mode not in ("hybrid", "dense", "bm25"):
            raise ValueError(f"Invalid mode '{mode}'. Choose from: 'hybrid', 'dense', 'bm25'.")

        self.embedder = embedder
        self.mode = mode
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        self._bm25: Optional[BM25Okapi] = None
        if mode in ("hybrid", "bm25"):
            self._build_bm25()

    def search(
        self,
        query: str,
        top_n: int = 10,
        mode: Optional[str] = None,
        filter_fn: Optional[callable] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search the indexed corpus.

        Args:
            query: Search query text.
            top_n: Number of results to return.
            mode: Override the default search mode for this query.
            filter_fn: Optional function that takes (text, metadata) and returns True to include.
            filters: Optional metadata filters passed to the vector store (store-specific).

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        search_mode = mode or self.mode

        if search_mode == "dense":
            results = self._dense_search(query, top_n * 3, filters=filters)
        elif search_mode == "bm25":
            results = self._bm25_search(query, top_n * 3)
        else:
            results = self._hybrid_search(query, top_n * 3, filters=filters)

        if filter_fn:
            results = [r for r in results if filter_fn(r.text, r.metadata)]

        return results[:top_n]

    def _dense_search(
        self,
        query: str,
        top_n: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Pure semantic similarity search via vector store."""
        query_embedding = self.embedder.encode(query)
        n = min(top_n, self.embedder.corpus_size)

        if n == 0:
            return []

        store_results = self.embedder._store.search(
            query_vector=query_embedding[0],
            k=n,
            filters=filters,
        )

        results = []
        for item in store_results:
            idx = int(item["id"])
            if idx < len(self.embedder._corpus):
                results.append(SearchResult(
                    index=idx,
                    text=self.embedder._corpus[idx],
                    score=float(item["score"]),
                    metadata=item.get("metadata", {}),
                ))
        return results

    def _bm25_search(self, query: str, top_n: int) -> List[SearchResult]:
        """Pure keyword search using BM25."""
        if self._bm25 is None:
            self._build_bm25()

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(SearchResult(
                index=int(idx),
                text=self.embedder._corpus[idx],
                score=float(scores[idx]),
                metadata=self.embedder._metadata[idx] if self.embedder._metadata else {},
            ))
        return results

    def _hybrid_search(
        self,
        query: str,
        top_n: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Hybrid search using Reciprocal Rank Fusion of dense + BM25 results."""
        dense_results = self._dense_search(query, top_n, filters=filters)
        bm25_results = self._bm25_search(query, top_n)

        rrf_scores: Dict[int, float] = {}
        index_map: Dict[int, SearchResult] = {}

        for rank, result in enumerate(dense_results):
            rrf_scores[result.index] = rrf_scores.get(result.index, 0.0)
            rrf_scores[result.index] += self.dense_weight / (self.rrf_k + rank + 1)
            index_map[result.index] = result

        for rank, result in enumerate(bm25_results):
            rrf_scores[result.index] = rrf_scores.get(result.index, 0.0)
            rrf_scores[result.index] += self.bm25_weight / (self.rrf_k + rank + 1)
            if result.index not in index_map:
                index_map[result.index] = result

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for idx in sorted_indices[:top_n]:
            result = index_map[idx]
            result.score = rrf_scores[idx]
            results.append(result)
        return results

    def _build_bm25(self) -> None:
        """Build BM25 index from corpus."""
        tokenized_corpus = [doc.lower().split() for doc in self.embedder._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)
