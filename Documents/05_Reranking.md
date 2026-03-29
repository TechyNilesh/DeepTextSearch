# Reranking

## Overview

Reranking is a critical second stage in modern search pipelines. While bi-encoder embeddings are fast, cross-encoder rerankers are much more accurate because they jointly encode the query and passage together.

**Pipeline**: Retrieve 20-100 candidates with `TextSearch` → Rerank top results with `Reranker` → Return top 5-10.

## Basic Usage

```python
from DeepTextSearch import TextEmbedder, TextSearch, Reranker

embedder = TextEmbedder()
embedder.index(your_corpus)
search = TextSearch(embedder)

# Retrieve candidates
results = search.search("deep learning", top_n=20)

# Rerank for precision
reranker = Reranker()
final = reranker.rerank_search_results("deep learning", results, top_n=5)
```

## Model Presets

### Tiny Models (< 50 MB) — CPU / Serverless / Edge

| Model | Size | Description |
|-------|------|-------------|
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | ~17 MB | Smallest reranker. Ultra fast |
| `cross-encoder/ms-marco-MiniLM-L-2-v2` | ~17 MB | Tiny 2-layer. Minimal latency |
| `cross-encoder/ms-marco-MiniLM-L-4-v2` | ~45 MB | Small 4-layer |

### Standard Models

| Model | Size | Description |
|-------|------|-------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90 MB | **Default.** Best speed/quality |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~130 MB | Higher quality |

### Large / Multilingual Models

| Model | Size | Languages | Description |
|-------|------|-----------|-------------|
| `BAAI/bge-reranker-v2-m3` | ~2.2 GB | 100+ | Best multilingual |
| `BAAI/bge-reranker-base` | ~1.1 GB | English | Balanced |
| `BAAI/bge-reranker-large` | ~1.3 GB | English | Highest accuracy |
| `jinaai/jina-reranker-v2-base-multilingual` | ~1.1 GB | 100+ | Jina multilingual |

## Choosing a Model

```python
# Tiny — for CPU, serverless, or edge deployments
reranker = Reranker("cross-encoder/ms-marco-TinyBERT-L-2-v2")

# Default — balanced speed and quality
reranker = Reranker()

# Multilingual — for non-English content
reranker = Reranker("BAAI/bge-reranker-v2-m3")

# Maximum accuracy — for quality-critical applications
reranker = Reranker("BAAI/bge-reranker-large")
```

## Three Ways to Rerank

### 1. Rerank SearchResult Objects

```python
results = search.search("query", top_n=20)
reranked = reranker.rerank_search_results("query", results, top_n=5)
# Returns: [{"text": ..., "score": ..., "index": ..., "metadata": ...}]
```

### 2. Rerank Plain Text Strings

```python
reranked = reranker.rerank_texts(
    "query",
    ["text 1", "text 2", "text 3"],
    top_n=2,
)
# Returns: [{"text": ..., "score": ...}]
```

### 3. Rerank with RerankRequest

```python
from DeepTextSearch import RerankRequest

request = RerankRequest(
    query="efficient search",
    passages=[
        {"text": "FAISS enables fast search.", "id": 1, "source": "docs"},
        {"text": "PostgreSQL has full-text search.", "id": 2, "source": "blog"},
    ],
)
reranked = reranker.rerank(request, top_n=1)
# Returns: [{"text": ..., "score": ..., "id": ..., "source": ...}]
# All original fields are preserved, "score" is added
```

## Custom / Fine-tuned Rerankers

```python
# Local fine-tuned model
reranker = Reranker("./my-fine-tuned-reranker")

# Private HuggingFace model
reranker = Reranker("your-org/your-reranker")
```

## Listing Available Presets

```python
from DeepTextSearch import RERANKER_PRESETS

for name, info in RERANKER_PRESETS.items():
    print(f"{name}: {info['description']} ({info['size_mb']} MB)")
```
