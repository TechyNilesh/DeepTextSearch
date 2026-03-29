# API Reference

Complete API documentation for all DeepTextSearch classes, methods, and parameters.

---

## `TextEmbedder`

The core class for embedding text and managing vector indexes.

### Constructor

```python
TextEmbedder(
    model_name: str = "BAAI/bge-m3",
    vector_store: Union[str, BaseVectorStore] = "faiss",
    store_config: Optional[dict] = None,
    index_dir: str = ".deeptextsearch",
    index_type: str = "flat",
    device: Optional[str] = None,
    batch_size: int = 64,
    normalize: bool = True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"BAAI/bge-m3"` | Any sentence-transformers HuggingFace model name, preset key, or local path to a saved model directory. |
| `vector_store` | `str` or `BaseVectorStore` | `"faiss"` | Vector store backend. String options: `"faiss"`, `"chroma"`, `"qdrant"`, `"postgres"`, `"mongo"`. Or pass a custom `BaseVectorStore` instance. |
| `store_config` | `dict` or `None` | `None` | Backend-specific configuration dict. Keys depend on the chosen backend (see [Vector Stores](04_Vector_Stores.md)). |
| `index_dir` | `str` | `".deeptextsearch"` | Directory path for saving/loading index data. Used as default persist location. |
| `index_type` | `str` | `"flat"` | FAISS index type: `"flat"` (exact), `"ivf"` (approximate), `"hnsw"` (approximate). Only used when `vector_store="faiss"`. |
| `device` | `str` or `None` | `None` | Inference device: `"cpu"`, `"cuda"`, `"mps"`. `None` auto-detects (CUDA > MPS > CPU). |
| `batch_size` | `int` | `64` | Batch size for encoding text. Larger = faster but more memory. |
| `normalize` | `bool` | `True` | Whether to L2-normalize embeddings. Required for cosine similarity with FAISS inner product index. |

**`store_config` keys by backend:**

| Backend | Key | Type | Default | Description |
|---------|-----|------|---------|-------------|
| `faiss` | `index_type` | `str` | `"flat"` | Override FAISS index type |
| `chroma` | `collection_name` | `str` | `"deep_text_search"` | ChromaDB collection name |
| `chroma` | `persist_directory` | `str` | `index_dir` | Storage directory |
| `qdrant` | `collection_name` | `str` | `"deep_text_search"` | Qdrant collection name |
| `qdrant` | `location` | `str` | `None` | Remote server URL (e.g. `"http://localhost:6333"`) |
| `qdrant` | `path` | `str` | `index_dir` | Local persistent storage path |
| `postgres` | `connection_string` | `str` | `"postgresql://localhost:5432/deeptextsearch"` | PostgreSQL connection string |
| `postgres` | `table_name` | `str` | `"text_vectors"` | Table name |
| `postgres` | `metadata_schema` | `dict` | `None` | `{"column_name": "SQL_TYPE"}` for dedicated indexed columns |
| `mongo` | `connection_string` | `str` | `"mongodb://localhost:27017"` | MongoDB connection string |
| `mongo` | `database_name` | `str` | `"deeptextsearch"` | Database name |
| `mongo` | `collection_name` | `str` | `"text_vectors"` | Collection name |
| `mongo` | `index_name` | `str` | `"vector_index"` | Atlas Vector Search index name |
| `mongo` | `metadata_fields` | `list[str]` | `None` | Fields to store as top-level document fields |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `.corpus_size` | `int` | Number of indexed documents. |
| `.dimension` | `int` | Embedding vector dimension. |
| `.model_name` | `str` | Resolved HuggingFace model name. |
| `.device` | `str` | Active inference device. |

### Methods

#### `.index(corpus, text_column=None, metadata_columns=None) → TextEmbedder`

Embed and index an entire text corpus. Returns `self` for method chaining.

| Parameter | Type | Description |
|-----------|------|-------------|
| `corpus` | `list[str]`, `pd.Series`, `pd.DataFrame` | The text corpus to index. |
| `text_column` | `str` or `None` | Required when `corpus` is a DataFrame. Column containing text. |
| `metadata_columns` | `list[str]` or `None` | DataFrame columns to store as metadata (DataFrame only). |

```python
embedder.index(["text 1", "text 2"])
embedder.index(df, text_column="content", metadata_columns=["title", "date"])
```

---

#### `.add(texts, metadata=None) → TextEmbedder`

Add texts to an existing index incrementally. Returns `self` for chaining.

| Parameter | Type | Description |
|-----------|------|-------------|
| `texts` | `str` or `list[str]` | Text(s) to add. |
| `metadata` | `dict`, `list[dict]`, or `None` | Metadata for the new texts. |

```python
embedder.add("New document.")
embedder.add(["Doc A", "Doc B"], metadata=[{"src": "web"}, {"src": "book"}])
```

---

#### `.delete(indices) → None`

Delete documents from the index by their corpus indices.

| Parameter | Type | Description |
|-----------|------|-------------|
| `indices` | `list[int]` | Corpus indices to delete. |

```python
embedder.delete([0, 5, 12])
```

---

#### `.encode(texts) → np.ndarray`

Encode text(s) into embedding vectors without adding to the index.

| Parameter | Type | Description |
|-----------|------|-------------|
| `texts` | `str` or `list[str]` | Text(s) to encode. |

**Returns:** `np.ndarray` of shape `(N, D)` where D is the embedding dimension.

```python
vectors = embedder.encode("Hello world")  # shape: (1, 1024)
vectors = embedder.encode(["A", "B"])     # shape: (2, 1024)
```

---

#### `.save(index_dir=None) → None`

Save index, corpus, and configuration to disk.

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_dir` | `str` or `None` | Override save directory. Uses `self.index_dir` if `None`. |

```python
embedder.save("/path/to/my_index")
```

---

#### `.load(index_dir, device=None) → TextEmbedder` *(classmethod)*

Load a previously saved index from disk.

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_dir` | `str` | Directory containing saved index files. |
| `device` | `str` or `None` | Override device for inference. |

**Returns:** `TextEmbedder` instance with loaded index.

```python
embedder = TextEmbedder.load("/path/to/my_index")
embedder = TextEmbedder.load("/path/to/my_index", device="cuda")
```

---

#### `.from_csv(file_path, text_column, ...) → TextEmbedder` *(classmethod)*

Load CSV, embed, and return a ready-to-search instance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to CSV file. |
| `text_column` | `str` | Column containing text. |
| `metadata_columns` | `list[str]` or `None` | Columns to store as metadata. |
| `model_name` | `str` | Embedding model name. |
| `**kwargs` | | Additional arguments passed to the constructor. |

```python
embedder = TextEmbedder.from_csv("data.csv", text_column="content")
```

---

## `TextSearch`

Hybrid search engine combining dense and BM25 retrieval.

### Constructor

```python
TextSearch(
    embedder: TextEmbedder,
    mode: str = "hybrid",
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
    rrf_k: int = 60,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedder` | `TextEmbedder` | — | A TextEmbedder with an indexed corpus. |
| `mode` | `str` | `"hybrid"` | Default search mode: `"hybrid"`, `"dense"`, or `"bm25"`. |
| `bm25_weight` | `float` | `0.4` | Weight for BM25 scores in hybrid RRF fusion. |
| `dense_weight` | `float` | `0.6` | Weight for dense scores in hybrid RRF fusion. |
| `rrf_k` | `int` | `60` | Reciprocal Rank Fusion constant. Lower = more weight to top ranks. |

### Methods

#### `.search(query, top_n=10, mode=None, filter_fn=None, filters=None) → list[SearchResult]`

Search the indexed corpus.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | — | Search query text. |
| `top_n` | `int` | `10` | Number of results to return. |
| `mode` | `str` or `None` | `None` | Override default mode for this query. |
| `filter_fn` | `callable` or `None` | `None` | Function `(text: str, metadata: dict) → bool`. Returns `True` to include. |
| `filters` | `dict` or `None` | `None` | Metadata filters passed directly to the vector store. |

**Returns:** `list[SearchResult]` sorted by relevance.

```python
results = search.search("machine learning", top_n=5)
results = search.search("exact term", mode="bm25")
results = search.search("AI", filters={"category": "tech"})
results = search.search("AI", filter_fn=lambda t, m: m.get("year", 0) >= 2023)
```

---

## `SearchResult`

A single search result returned by `TextSearch.search()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.index` | `int` | Position in the original corpus. |
| `.text` | `str` | The matched text. |
| `.score` | `float` | Relevance score. |
| `.metadata` | `dict` | Associated metadata. |

### Methods

#### `.to_dict() → dict`

Convert to a plain dictionary.

```python
{"index": 0, "text": "...", "score": 0.9523, "metadata": {"category": "AI"}}
```

---

## `Reranker`

Cross-encoder reranker for search result refinement.

### Constructor

```python
Reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length: int = 512,
    device: Optional[str] = None,
    batch_size: int = 64,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Any HuggingFace cross-encoder model or local path. |
| `max_length` | `int` | `512` | Maximum token length for query-passage pairs. |
| `device` | `str` or `None` | `None` | Inference device. `None` auto-detects. |
| `batch_size` | `int` | `64` | Batch size for scoring. |

### Methods

#### `.rerank(request, top_n=None) → list[dict]`

Rerank passages by relevance to a query.

| Parameter | Type | Description |
|-----------|------|-------------|
| `request` | `RerankRequest` | Query and passages to rerank. |
| `top_n` | `int` or `None` | Return only top N. `None` returns all. |

**Returns:** List of passage dicts sorted by relevance, with `"score"` added. All original fields are preserved.

---

#### `.rerank_texts(query, texts, top_n=None) → list[dict]`

Rerank plain text strings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query. |
| `texts` | `list[str]` | Texts to rerank. |
| `top_n` | `int` or `None` | Return only top N. |

**Returns:** `[{"text": "...", "score": 0.95}, ...]`

---

#### `.rerank_search_results(query, search_results, top_n=None) → list[dict]`

Rerank `SearchResult` objects from `TextSearch.search()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query. |
| `search_results` | `list[SearchResult]` | Results from `TextSearch.search()`. |
| `top_n` | `int` or `None` | Return only top N. |

**Returns:** `[{"text": "...", "score": 0.95, "index": 3, "metadata": {...}}, ...]`

---

## `RerankRequest`

Container for a reranking request.

```python
RerankRequest(query: str, passages: list[dict])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | The search query. |
| `passages` | `list[dict]` | List of passage dicts. Each must have a `"text"` key. May include any other fields (`"id"`, `"source"`, etc.) — they will be preserved in the output. |

---

## Vector Stores

All vector stores implement the `BaseVectorStore` abstract interface.

### `BaseVectorStore` (Abstract)

```python
class BaseVectorStore(ABC):
    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict] = None) -> None: ...
    def search(self, query_vector: np.ndarray, k: int = 10, filters: dict = None) -> list[dict]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

**`.search()` return format:**
```python
[
    {"id": "0", "score": 0.95, "metadata": {"category": "AI"}},
    {"id": "1", "score": 0.87, "metadata": {"category": "ML"}},
]
```

### `FAISSStore`

```python
FAISSStore(dimension: int, index_type: str = "flat")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimension` | `int` | — | Vector dimension. |
| `index_type` | `str` | `"flat"` | `"flat"` (exact), `"ivf"` (approximate), `"hnsw"` (approximate). |

### `ChromaStore`

```python
ChromaStore(collection_name: str = "deep_text_search", persist_directory: str = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | `"deep_text_search"` | ChromaDB collection name. |
| `persist_directory` | `str` or `None` | `None` | Directory for persistent storage. `None` for in-memory. |

### `QdrantStore`

```python
QdrantStore(collection_name: str = "deep_text_search", location: str = None, path: str = None, dimension: int = 1024)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | `"deep_text_search"` | Qdrant collection name. |
| `location` | `str` or `None` | `None` | Remote server URL. |
| `path` | `str` or `None` | `None` | Local persistent storage path. |
| `dimension` | `int` | `1024` | Vector dimension. |

### `PostgresStore`

```python
PostgresStore(connection_string: str, table_name: str = "text_vectors", dimension: int = 1024, metadata_schema: dict = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `connection_string` | `str` | `"postgresql://localhost:5432/deeptextsearch"` | PostgreSQL connection string. |
| `table_name` | `str` | `"text_vectors"` | Table name. |
| `dimension` | `int` | `1024` | Vector dimension. |
| `metadata_schema` | `dict` or `None` | `None` | `{"column_name": "SQL_TYPE"}` for dedicated indexed columns alongside JSONB metadata. |

### `MongoStore`

```python
MongoStore(connection_string: str, database_name: str = "deeptextsearch", collection_name: str = "text_vectors", index_name: str = "vector_index", dimension: int = 1024, metadata_fields: list[str] = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `connection_string` | `str` | `"mongodb://localhost:27017"` | MongoDB connection string. |
| `database_name` | `str` | `"deeptextsearch"` | Database name. |
| `collection_name` | `str` | `"text_vectors"` | Collection name. |
| `index_name` | `str` | `"vector_index"` | Atlas Vector Search index name. |
| `dimension` | `int` | `1024` | Vector dimension. |
| `metadata_fields` | `list[str]` or `None` | `None` | Fields to store as top-level document fields for native MongoDB indexing. |

---

## Agent Tools

### `TextSearchTool`

Generic callable tool for AI agent frameworks.

```python
TextSearchTool(embedder: TextEmbedder, mode: str = "hybrid", reranker: Reranker = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedder` | `TextEmbedder` | — | Embedder with indexed corpus. |
| `mode` | `str` | `"hybrid"` | Default search mode. |
| `reranker` | `Reranker` or `None` | `None` | Optional reranker for result refinement. |

**Calling:**
```python
result_json = tool("query", k=5, mode="auto")  # Returns JSON string
```

**Properties:**
- `.tool_definition` → `dict` — OpenAI/Claude function-calling compatible JSON schema.

### `create_langchain_retriever()`

```python
create_langchain_retriever(
    embedder: TextEmbedder,
    mode: str = "hybrid",
    reranker: Reranker = None,
    top_n: int = 5,
) → BaseRetriever
```

Returns a LangChain `BaseRetriever` instance. Requires `langchain-core`.

### `create_llamaindex_retriever()`

```python
create_llamaindex_retriever(
    embedder: TextEmbedder,
    mode: str = "hybrid",
    reranker: Reranker = None,
    top_n: int = 5,
) → BaseRetriever
```

Returns a LlamaIndex `BaseRetriever` instance. Requires `llama-index-core`.

---

## Configuration

### `EMBEDDING_PRESETS`

`dict` — Recommended embedding models with metadata.

```python
from DeepTextSearch import EMBEDDING_PRESETS
for name, info in EMBEDDING_PRESETS.items():
    print(f"{name}: {info['dimensions']}d, ~{info['size_mb']} MB, {info['languages']}")
```

### `RERANKER_PRESETS`

`dict` — Recommended reranker models with metadata.

```python
from DeepTextSearch import RERANKER_PRESETS
for name, info in RERANKER_PRESETS.items():
    print(f"{name}: ~{info['size_mb']} MB, {info['languages']}")
```

### `get_device(device=None) → str`

Auto-detect best available device. Returns `"cuda"`, `"mps"`, or `"cpu"`.

```python
from DeepTextSearch.config import get_device
device = get_device()       # auto-detect
device = get_device("cpu")  # force CPU
```
