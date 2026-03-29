# Search Engine

## Overview

`TextSearch` provides three search modes that can be mixed per query:

| Mode | How It Works | Best For |
|------|-------------|----------|
| `"hybrid"` (default) | Dense + BM25 combined with Reciprocal Rank Fusion | General-purpose, best overall accuracy |
| `"dense"` | Pure semantic similarity via vector store | Conceptual/meaning-based queries |
| `"bm25"` | Pure keyword matching via BM25 | Exact phrase/keyword queries |

## Basic Usage

```python
from DeepTextSearch import TextEmbedder, TextSearch

embedder = TextEmbedder()
embedder.index(your_corpus)

search = TextSearch(embedder)
results = search.search("your query", top_n=10)
```

## Search Modes

### Hybrid Search (Default)

Combines dense semantic search and BM25 keyword search using Reciprocal Rank Fusion (RRF). This consistently outperforms either method alone.

```python
search = TextSearch(embedder, mode="hybrid")
results = search.search("machine learning applications")
```

### Dense Search

Pure semantic similarity — finds conceptually related texts even without keyword overlap.

```python
results = search.search("ML use cases", mode="dense")
# Will match "machine learning applications" even though no keywords overlap
```

### BM25 Search

Pure keyword matching — best for exact phrases or specific terms.

```python
results = search.search("PostgreSQL indexing", mode="bm25")
```

### Per-Query Mode Override

```python
search = TextSearch(embedder, mode="hybrid")  # default
results = search.search("specific error code XYZ-123", mode="bm25")  # override for this query
```

## Tuning Hybrid Search

Control the balance between dense and BM25:

```python
# Favor semantic understanding (default)
search = TextSearch(embedder, dense_weight=0.6, bm25_weight=0.4)

# Favor keyword matching
search = TextSearch(embedder, dense_weight=0.3, bm25_weight=0.7)

# Pure semantic with RRF structure
search = TextSearch(embedder, dense_weight=1.0, bm25_weight=0.0)
```

The `rrf_k` parameter controls the Reciprocal Rank Fusion constant (default 60). Lower values give more weight to top-ranked results:

```python
search = TextSearch(embedder, rrf_k=30)  # More aggressive ranking
```

## Metadata Filtering

### Function-based Filtering

```python
# Filter by metadata after search
results = search.search(
    "machine learning",
    top_n=10,
    filter_fn=lambda text, meta: meta.get("category") == "AI",
)

# Multiple conditions
results = search.search(
    "python tutorial",
    top_n=10,
    filter_fn=lambda text, meta: (
        meta.get("language") == "en"
        and meta.get("year", 0) >= 2023
    ),
)
```

### Store-level Filtering

Pass filters directly to the vector store (more efficient for large corpora):

```python
results = search.search(
    "machine learning",
    top_n=10,
    filters={"category": "AI"},  # passed to vector store
)
```

## SearchResult Object

Each result has these attributes:

```python
result = results[0]
result.index      # int: position in original corpus
result.text       # str: the matched text
result.score      # float: relevance score
result.metadata   # dict: associated metadata

# Convert to dict
result.to_dict()  # {"index": 0, "text": "...", "score": 0.95, "metadata": {...}}
```
