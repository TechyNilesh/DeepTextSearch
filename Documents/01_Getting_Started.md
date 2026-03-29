# Getting Started

## Installation

**From PyPI:**
```bash
pip install DeepTextSearch
```

**From GitHub (latest):**
```bash
pip install git+https://github.com/TechyNilesh/DeepTextSearch.git
```

**Optional extras:**
```bash
# Vector store backends
pip install 'DeepTextSearch[chroma]'      # ChromaDB
pip install 'DeepTextSearch[qdrant]'      # Qdrant
pip install 'DeepTextSearch[postgres]'    # PostgreSQL + pgvector
pip install 'DeepTextSearch[mongo]'       # MongoDB Atlas

# Integrations
pip install 'DeepTextSearch[mcp]'         # MCP server
pip install 'DeepTextSearch[langchain]'   # LangChain retriever
pip install 'DeepTextSearch[llamaindex]'  # LlamaIndex retriever
pip install 'DeepTextSearch[gpu]'         # FAISS GPU

# Everything
pip install 'DeepTextSearch[all]'
```

## Quick Start

```python
from DeepTextSearch import TextEmbedder, TextSearch

# 1. Create an embedder and index your corpus
embedder = TextEmbedder()
embedder.index([
    "Python is a versatile programming language.",
    "Machine learning is transforming industries.",
    "FAISS enables fast similarity search.",
    "Natural language processing understands text.",
])

# 2. Search
search = TextSearch(embedder)
results = search.search("What is NLP?", top_n=3)

for r in results:
    print(f"[{r.score:.4f}] {r.text}")
```

## Core Concepts

DeepTextSearch has four core components:

### 1. TextEmbedder
Converts text into dense vector embeddings using transformer models and stores them in a vector database.

```python
embedder = TextEmbedder(
    model_name="BAAI/bge-m3",       # Any HuggingFace model
    vector_store="faiss",            # faiss, chroma, qdrant, postgres, mongo
    index_dir=".deeptextsearch",     # Where to save the index
    device=None,                     # cpu, cuda, mps, or auto-detect
)
```

### 2. TextSearch
Searches the indexed corpus using hybrid (dense + BM25), dense-only, or BM25-only modes.

```python
search = TextSearch(
    embedder,
    mode="hybrid",       # hybrid, dense, or bm25
    dense_weight=0.6,    # Weight for dense scores
    bm25_weight=0.4,     # Weight for BM25 scores
)
results = search.search("your query", top_n=10)
```

### 3. Reranker
Refines search results using a cross-encoder model for higher accuracy.

```python
from DeepTextSearch import Reranker
reranker = Reranker()
reranked = reranker.rerank_search_results("query", results, top_n=5)
```

### 4. Vector Stores
Pluggable backends for storing and searching vectors. FAISS is the default, but you can use ChromaDB, Qdrant, PostgreSQL, or MongoDB.

## Typical Workflow

```python
from DeepTextSearch import TextEmbedder, TextSearch, Reranker

# Step 1: Index
embedder = TextEmbedder()
embedder.index(your_texts)
embedder.save("./my_index")

# Step 2: Search
search = TextSearch(embedder)
results = search.search("your query", top_n=20)

# Step 3: Rerank (optional but recommended)
reranker = Reranker()
final = reranker.rerank_search_results("your query", results, top_n=5)

# Step 4: Use results
for r in final:
    print(f"[{r['score']:.4f}] {r['text']}")
```

## Next Steps

- [Embedding Models](02_Embeddings.md) — choosing and configuring models
- [Search Engine](03_Search_Engine.md) — hybrid, dense, and BM25 search
- [Vector Stores](04_Vector_Stores.md) — FAISS, ChromaDB, Qdrant, PostgreSQL, MongoDB
- [Reranking](05_Reranking.md) — cross-encoder models for precision
- [Agentic Integration](06_Agentic_Integration.md) — MCP, LangChain, LlamaIndex
- [Data Loading](07_Data_Loading.md) — CSV, DataFrame, and list input
- [API Reference](08_API_Reference.md) — complete API documentation
