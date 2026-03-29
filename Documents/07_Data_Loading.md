# Data Loading

## Overview

DeepTextSearch accepts text data from multiple sources:
- Python lists
- Pandas Series
- Pandas DataFrames
- CSV files

## From a Python List

```python
from DeepTextSearch import TextEmbedder

corpus = [
    "First document text.",
    "Second document text.",
    "Third document text.",
]

embedder = TextEmbedder()
embedder.index(corpus)
```

## From a CSV File

One-line convenience method:

```python
embedder = TextEmbedder.from_csv(
    "articles.csv",
    text_column="content",
    metadata_columns=["title", "date", "category"],
)
```

## From a Pandas DataFrame

```python
import pandas as pd

df = pd.read_csv("articles.csv")

embedder = TextEmbedder()
embedder.index(
    df,
    text_column="content",
    metadata_columns=["title", "author", "date"],
)
```

## From a Pandas Series

```python
texts = df["content"]
embedder = TextEmbedder()
embedder.index(texts)
```

## Metadata

Metadata is stored alongside each document and is available in search results:

```python
embedder.index(
    df,
    text_column="content",
    metadata_columns=["title", "category", "date"],
)

search = TextSearch(embedder)
results = search.search("query")
print(results[0].metadata)  # {"title": "...", "category": "...", "date": "..."}
```

## Incremental Indexing

Add documents to an existing index:

```python
# Add a single document
embedder.add("New document text.")

# Add with metadata
embedder.add(
    "Another document.",
    metadata={"source": "web"},
)

# Add multiple documents
embedder.add(
    ["Doc 1", "Doc 2", "Doc 3"],
    metadata=[
        {"source": "web"},
        {"source": "paper"},
        {"source": "book"},
    ],
)
```

## Deleting Documents

```python
# Delete by corpus index
embedder.delete([0, 5, 12])
```

## Saving and Loading

Save to any local folder path:

```python
# Save
embedder.save("/path/to/my_index")
embedder.save("./projects/search_v2")
embedder.save("~/indexes/production")

# Load (no re-embedding needed)
embedder = TextEmbedder.load("/path/to/my_index")
```

Saved files:
- `config.json` — model name, dimension, settings
- `corpus.json` — all indexed texts
- `index.faiss` + `store_meta.json` — vector index and metadata (FAISS)
- Other stores persist automatically via their own mechanisms
