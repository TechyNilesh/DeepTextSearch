# Vector Stores

## Overview

DeepTextSearch supports pluggable vector store backends. Pass a string shorthand or a custom `BaseVectorStore` instance to `TextEmbedder`.

| Backend | Install | Persistence | Best For |
|---------|---------|-------------|----------|
| **FAISS** (default) | Included | Local files | Maximum speed, embedded use, prototyping |
| **ChromaDB** | `pip install 'DeepTextSearch[chroma]'` | Local files | Simple persistent storage, small-medium corpora |
| **Qdrant** | `pip install 'DeepTextSearch[qdrant]'` | Local/remote | Production, metadata filtering, self-hosted |
| **PostgreSQL** | `pip install 'DeepTextSearch[postgres]'` | Database | Enterprise, existing Postgres infrastructure |
| **MongoDB** | `pip install 'DeepTextSearch[mongo]'` | Database | MongoDB Atlas, document-oriented workflows |

## FAISS (Default)

Zero-config local vector store. Supports three index types:

| Index Type | Speed | Accuracy | Memory | Best For |
|-----------|-------|----------|--------|----------|
| `"flat"` (default) | O(n) | Exact | High | Small-medium corpora (< 1M docs) |
| `"hnsw"` | Very fast | Approximate | High | Large corpora, fast queries |
| `"ivf"` | Fast | Approximate | Medium | Large corpora, balanced |

```python
# Flat index (exact search)
embedder = TextEmbedder(vector_store="faiss", index_type="flat")

# HNSW index (fast approximate search)
embedder = TextEmbedder(vector_store="faiss", index_type="hnsw")

# IVF index (approximate, good for large corpora)
embedder = TextEmbedder(vector_store="faiss", index_type="ivf")
```

### Save & Load

```python
embedder.save("/path/to/index")
embedder = TextEmbedder.load("/path/to/index")
```

## ChromaDB

Persistent local vector store with built-in metadata filtering.

```bash
pip install 'DeepTextSearch[chroma]'
```

```python
# ChromaDB with persistent storage
embedder = TextEmbedder(vector_store="chroma", index_dir="./my_chroma_db")
embedder.index(your_corpus)
# ChromaDB auto-persists — no manual save needed
```

## Qdrant

Production-grade vector database. Supports local, in-memory, and remote server modes.

```bash
pip install 'DeepTextSearch[qdrant]'
```

```python
# Local persistent storage
embedder = TextEmbedder(vector_store="qdrant", index_dir="./my_qdrant_db")

# Remote Qdrant server
from DeepTextSearch.vectorstores import QdrantStore
store = QdrantStore(
    collection_name="my_texts",
    location="http://localhost:6333",
    dimension=1024,
)
embedder = TextEmbedder(vector_store=store)
```

## PostgreSQL (pgvector)

Use your existing PostgreSQL database for vector storage.

```bash
pip install 'DeepTextSearch[postgres]'
```

Requires the `pgvector` extension installed in your PostgreSQL database:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Basic Usage

```python
# Via store_config (recommended)
embedder = TextEmbedder(
    vector_store="postgres",
    store_config={
        "connection_string": "postgresql://user:password@localhost:5432/mydb",
        "table_name": "article_vectors",
    },
)
embedder.index(your_corpus)
```

### With Dedicated Metadata Columns

Define a `metadata_schema` to create dedicated SQL columns for metadata fields.
This enables native PostgreSQL indexing and fast filtering:

```python
embedder = TextEmbedder(
    vector_store="postgres",
    store_config={
        "connection_string": "postgresql://user:pass@localhost/mydb",
        "table_name": "article_vectors",
        "metadata_schema": {
            "category": "TEXT",
            "language": "TEXT",
            "year": "INTEGER",
            "source": "TEXT",
        },
    },
)

# Index with metadata — schema fields get their own SQL columns
import pandas as pd
df = pd.read_csv("articles.csv")
embedder.index(df, text_column="content", metadata_columns=["category", "language", "year", "source"])

# Filter on dedicated columns (uses SQL index, very fast)
search = TextSearch(embedder)
results = search.search("AI research", filters={"category": "tech", "year": 2024})
```

### Direct Store Usage

```python
from DeepTextSearch.vectorstores import PostgresStore

store = PostgresStore(
    connection_string="postgresql://user:password@localhost:5432/mydb",
    table_name="article_vectors",
    dimension=1024,
    metadata_schema={"category": "TEXT", "author": "TEXT"},
)
embedder = TextEmbedder(vector_store=store)
```

## MongoDB Atlas

Use MongoDB Atlas Vector Search for cloud-native vector storage.

```bash
pip install 'DeepTextSearch[mongo]'
```

Requires a [Vector Search index](https://www.mongodb.com/docs/atlas/atlas-vector-search/) configured on your collection.

### Basic Usage

```python
embedder = TextEmbedder(
    vector_store="mongo",
    store_config={
        "connection_string": "mongodb+srv://user:pass@cluster.mongodb.net",
        "database_name": "search_db",
        "collection_name": "articles",
        "index_name": "article_vector_index",
    },
)
embedder.index(your_corpus)
```

### With Top-Level Metadata Fields

Specify `metadata_fields` to store certain metadata as top-level MongoDB document
fields. This enables native MongoDB indexing and querying:

```python
embedder = TextEmbedder(
    vector_store="mongo",
    store_config={
        "connection_string": "mongodb+srv://user:pass@cluster.mongodb.net",
        "database_name": "search_db",
        "collection_name": "articles",
        "metadata_fields": ["category", "language", "author", "year"],
    },
)

# Index with metadata — specified fields become top-level document fields
embedder.index(df, text_column="content", metadata_columns=["category", "language", "author", "year"])

# Filter on top-level fields (uses MongoDB index)
search = TextSearch(embedder)
results = search.search("machine learning", filters={"category": "AI"})
```

### Direct Store Usage

```python
from DeepTextSearch.vectorstores import MongoStore

store = MongoStore(
    connection_string="mongodb+srv://user:pass@cluster.mongodb.net",
    database_name="search_db",
    collection_name="articles",
    index_name="vector_index",
    dimension=1024,
    metadata_fields=["category", "author"],
)
embedder = TextEmbedder(vector_store=store)
```

## Custom Vector Store

Extend `BaseVectorStore` to use any backend:

```python
from DeepTextSearch import BaseVectorStore
import numpy as np

class MyCustomStore(BaseVectorStore):
    def add(self, ids, vectors, metadata=None):
        # Store vectors
        ...

    def search(self, query_vector, k=10, filters=None):
        # Return list of {"id": ..., "score": ..., "metadata": ...}
        ...

    def delete(self, ids):
        ...

    def count(self):
        ...

    def save(self, path):
        ...

    def load(self, path):
        ...

# Use it
embedder = TextEmbedder(vector_store=MyCustomStore())
```

## Vector Store Operations

All stores support these operations:

```python
store.add(ids, vectors, metadata)    # Add vectors
store.search(query_vector, k=10)     # Search
store.delete(ids)                    # Delete by ID
store.count()                        # Count vectors
store.save(path)                     # Persist to disk
store.load(path)                     # Load from disk
```
