"""DeepTextSearch - AI-powered multilingual text search and reranking engine.

A modern semantic search library with hybrid search (dense + BM25),
pluggable vector store backends, and cross-encoder reranking.

Basic usage:
    >>> from DeepTextSearch import TextEmbedder, TextSearch, Reranker
    >>> embedder = TextEmbedder()
    >>> embedder.index(["Hello world", "AI is the future", "Python rocks"])
    >>> search = TextSearch(embedder)
    >>> results = search.search("artificial intelligence")
"""

from .config import EMBEDDING_PRESETS, RERANKER_PRESETS
from .embedder import TextEmbedder
from .reranker import Reranker, RerankRequest
from .searcher import SearchResult, TextSearch
from .vectorstores import BaseVectorStore, FAISSStore

__version__ = "1.0.0"

__all__ = [
    # Core
    "TextEmbedder",
    "TextSearch",
    "SearchResult",
    "Reranker",
    "RerankRequest",
    # Vector stores
    "BaseVectorStore",
    "FAISSStore",
    # Presets
    "EMBEDDING_PRESETS",
    "RERANKER_PRESETS",
]
