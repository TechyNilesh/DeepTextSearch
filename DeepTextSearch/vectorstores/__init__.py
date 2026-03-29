"""Vector store backends for DeepTextSearch."""

from .base import BaseVectorStore
from .faiss_store import FAISSStore

__all__ = ["BaseVectorStore", "FAISSStore"]

# Optional backends — available only when their dependencies are installed
try:
    from .chroma_store import ChromaStore
    __all__.append("ChromaStore")
except ImportError:
    pass

try:
    from .qdrant_store import QdrantStore
    __all__.append("QdrantStore")
except ImportError:
    pass

try:
    from .postgres_store import PostgresStore
    __all__.append("PostgresStore")
except ImportError:
    pass

try:
    from .mongo_store import MongoStore
    __all__.append("MongoStore")
except ImportError:
    pass
