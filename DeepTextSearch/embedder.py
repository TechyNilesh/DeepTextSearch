"""Text embedding engine with pluggable vector store backends."""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_INDEX_DIR, get_device
from .vectorstores.base import BaseVectorStore


class TextEmbedder:
    """Embeds text corpus using sentence-transformers and stores in a vector database.

    Supports pluggable vector store backends:
    - "faiss" (default) — local FAISS index with flat/ivf/hnsw support
    - "chroma" — ChromaDB with persistent or in-memory storage
    - "qdrant" — Qdrant with local, in-memory, or remote server
    - Any custom BaseVectorStore instance

    Args:
        model_name: Any sentence-transformers compatible HuggingFace model or local path.
        vector_store: Store backend — "faiss" (default), "chroma", "qdrant",
            "postgres", "mongo", or a BaseVectorStore instance.
        store_config: Dict of backend-specific configuration passed to the vector store
            constructor. Keys depend on the backend. See examples below.
        index_dir: Directory to save/load index and corpus data.
        index_type: FAISS index type — "flat" (default), "ivf", or "hnsw".
            Only used when vector_store="faiss".
        device: Device for inference ('cpu', 'cuda', 'mps', or None for auto-detect).
        batch_size: Batch size for encoding.
        normalize: Whether to L2-normalize embeddings.

    Example:
        >>> embedder = TextEmbedder()
        >>> embedder.index(["Hello world", "AI is amazing", "Python rocks"])
        >>> embedder.save()

        # Use ChromaDB
        >>> embedder = TextEmbedder(vector_store="chroma")

        # Use Qdrant with custom collection
        >>> embedder = TextEmbedder(
        ...     vector_store="qdrant",
        ...     store_config={"collection_name": "my_articles"},
        ... )

        # Use PostgreSQL with connection string and custom table
        >>> embedder = TextEmbedder(
        ...     vector_store="postgres",
        ...     store_config={
        ...         "connection_string": "postgresql://user:pass@localhost/mydb",
        ...         "table_name": "article_embeddings",
        ...     },
        ... )

        # Use MongoDB with custom database and collection
        >>> embedder = TextEmbedder(
        ...     vector_store="mongo",
        ...     store_config={
        ...         "connection_string": "mongodb+srv://user:pass@cluster.mongodb.net",
        ...         "database_name": "search_db",
        ...         "collection_name": "articles",
        ...         "index_name": "article_vector_index",
        ...     },
        ... )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        vector_store: Union[str, BaseVectorStore] = "faiss",
        store_config: Optional[dict] = None,
        index_dir: str = DEFAULT_INDEX_DIR,
        index_type: str = "flat",
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_type = index_type
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = get_device(device)
        self._store_type = vector_store if isinstance(vector_store, str) else "custom"
        self._store_config = store_config or {}

        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._store: Optional[BaseVectorStore] = (
            vector_store if isinstance(vector_store, BaseVectorStore) else None
        )
        self._corpus: List[str] = []
        self._metadata: List[dict] = []

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def _create_store(self, dimension: int) -> BaseVectorStore:
        """Create a vector store instance based on the configured backend.

        User-provided store_config values override defaults.
        """
        cfg = self._store_config

        if isinstance(self._store_type, str) and self._store_type != "custom":
            if self._store_type == "faiss":
                from .vectorstores.faiss_store import FAISSStore
                return FAISSStore(
                    dimension=dimension,
                    index_type=cfg.get("index_type", self.index_type),
                )
            elif self._store_type == "chroma":
                from .vectorstores.chroma_store import ChromaStore
                return ChromaStore(
                    collection_name=cfg.get("collection_name", "deep_text_search"),
                    persist_directory=cfg.get("persist_directory", str(self.index_dir)),
                )
            elif self._store_type == "qdrant":
                from .vectorstores.qdrant_store import QdrantStore
                return QdrantStore(
                    collection_name=cfg.get("collection_name", "deep_text_search"),
                    location=cfg.get("location"),
                    path=cfg.get("path", str(self.index_dir)),
                    dimension=dimension,
                )
            elif self._store_type == "postgres":
                from .vectorstores.postgres_store import PostgresStore
                return PostgresStore(
                    connection_string=cfg.get(
                        "connection_string",
                        "postgresql://localhost:5432/deeptextsearch",
                    ),
                    table_name=cfg.get("table_name", "text_vectors"),
                    dimension=dimension,
                    metadata_schema=cfg.get("metadata_schema"),
                )
            elif self._store_type == "mongo":
                from .vectorstores.mongo_store import MongoStore
                return MongoStore(
                    connection_string=cfg.get(
                        "connection_string",
                        "mongodb://localhost:27017",
                    ),
                    database_name=cfg.get("database_name", "deeptextsearch"),
                    collection_name=cfg.get("collection_name", "text_vectors"),
                    index_name=cfg.get("index_name", "vector_index"),
                    dimension=dimension,
                    metadata_fields=cfg.get("metadata_fields"),
                )
            else:
                raise ValueError(
                    f"Unknown vector_store: '{self._store_type}'. "
                    "Choose from: 'faiss', 'chroma', 'qdrant', 'postgres', 'mongo', "
                    "or pass a BaseVectorStore instance."
                )
        return self._store

    def index(
        self,
        corpus: Union[List[str], pd.Series, pd.DataFrame],
        text_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
    ) -> "TextEmbedder":
        """Embed and index a text corpus.

        Args:
            corpus: List of texts, pandas Series, or DataFrame.
            text_column: Column name if corpus is a DataFrame.
            metadata_columns: Additional columns to store as metadata (DataFrame only).

        Returns:
            self (for chaining).
        """
        texts, metadata = self._prepare_corpus(corpus, text_column, metadata_columns)

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        embeddings = embeddings.astype(np.float32)

        self._store = self._create_store(embeddings.shape[1])
        ids = [str(i) for i in range(len(texts))]
        self._store.add(ids, embeddings, metadata)
        self._corpus = texts
        self._metadata = metadata

        return self

    def add(
        self,
        texts: Union[List[str], str],
        metadata: Optional[Union[List[dict], dict]] = None,
    ) -> "TextEmbedder":
        """Add new texts to an existing index (incremental indexing).

        Args:
            texts: Single text or list of texts to add.
            metadata: Optional metadata dict(s) for the new texts.

        Returns:
            self (for chaining).
        """
        if self._store is None:
            raise ValueError("No index exists. Call .index() first.")

        if isinstance(texts, str):
            texts = [texts]
        if metadata is None:
            metadata = [{} for _ in texts]
        elif isinstance(metadata, dict):
            metadata = [metadata]

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        embeddings = embeddings.astype(np.float32)

        start_id = len(self._corpus)
        ids = [str(start_id + i) for i in range(len(texts))]
        self._store.add(ids, embeddings, metadata)
        self._corpus.extend(texts)
        self._metadata.extend(metadata)

        return self

    def delete(self, indices: List[int]) -> None:
        """Delete texts from the index by their corpus indices.

        Args:
            indices: List of corpus indices to delete.
        """
        if self._store is None:
            raise ValueError("No index exists.")

        ids = [str(i) for i in indices]
        self._store.delete(ids)

        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self._corpus):
                self._corpus.pop(idx)
                self._metadata.pop(idx)

    def encode(self, texts: Union[List[str], str]) -> np.ndarray:
        """Encode text(s) into embeddings without indexing.

        Args:
            texts: Single text or list of texts.

        Returns:
            Numpy array of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings.astype(np.float32)

    def save(self, index_dir: Optional[str] = None) -> None:
        """Save index, corpus, and metadata to disk.

        Args:
            index_dir: Override the default save directory.
        """
        if self._store is None:
            raise ValueError("No index to save. Call .index() first.")

        save_dir = Path(index_dir) if index_dir else self.index_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        self._store.save(str(save_dir))

        with open(save_dir / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(self._corpus, f, ensure_ascii=False)

        config = {
            "model_name": self.model_name,
            "normalize": self.normalize,
            "corpus_size": len(self._corpus),
            "dimension": self.dimension,
            "store_type": self._store_type,
            "index_type": self.index_type,
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        index_dir: str = DEFAULT_INDEX_DIR,
        device: Optional[str] = None,
    ) -> "TextEmbedder":
        """Load a previously saved index from disk.

        Args:
            index_dir: Directory containing saved index files.
            device: Device for inference.

        Returns:
            TextEmbedder instance with loaded index.
        """
        load_dir = Path(index_dir)

        with open(load_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        store_type = config.get("store_type", "faiss")

        instance = cls(
            model_name=config["model_name"],
            vector_store=store_type,
            index_dir=index_dir,
            index_type=config.get("index_type", "flat"),
            device=device,
            normalize=config.get("normalize", True),
        )

        # Create and load the store
        instance._store = instance._create_store(config["dimension"])
        instance._store.load(str(load_dir))

        with open(load_dir / "corpus.json", "r", encoding="utf-8") as f:
            instance._corpus = json.load(f)

        # Metadata is stored inside the vector store now
        instance._metadata = [{} for _ in instance._corpus]

        return instance

    def _prepare_corpus(self, corpus, text_column, metadata_columns):
        """Convert various input formats to (texts, metadata) lists."""
        if isinstance(corpus, pd.DataFrame):
            if text_column is None:
                raise ValueError("text_column is required when corpus is a DataFrame.")
            texts = corpus[text_column].dropna().astype(str).tolist()
            metadata = []
            if metadata_columns:
                for _, row in corpus.iterrows():
                    if pd.notna(row[text_column]):
                        metadata.append({col: row[col] for col in metadata_columns if col in row.index})
            else:
                metadata = [{} for _ in texts]
        elif isinstance(corpus, pd.Series):
            texts = corpus.dropna().astype(str).tolist()
            metadata = [{} for _ in texts]
        elif isinstance(corpus, list):
            texts = [str(t) for t in corpus if t is not None]
            metadata = [{} for _ in texts]
        else:
            raise TypeError(f"Unsupported corpus type: {type(corpus)}. Use list, Series, or DataFrame.")

        if not texts:
            raise ValueError("Corpus is empty after filtering.")

        return texts, metadata

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        text_column: str,
        metadata_columns: Optional[List[str]] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        **kwargs,
    ) -> "TextEmbedder":
        """Convenience method: load CSV, embed, and return ready-to-search embedder.

        Args:
            file_path: Path to CSV file.
            text_column: Name of the column containing text.
            metadata_columns: Additional columns to store as metadata.
            model_name: Embedding model name.
            **kwargs: Additional arguments passed to TextEmbedder constructor.

        Returns:
            TextEmbedder instance with indexed corpus.
        """
        df = pd.read_csv(file_path)
        instance = cls(model_name=model_name, **kwargs)
        instance.index(df, text_column=text_column, metadata_columns=metadata_columns)
        return instance
