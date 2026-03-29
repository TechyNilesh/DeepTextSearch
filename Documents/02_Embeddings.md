# Embedding Models

## Overview

DeepTextSearch uses [sentence-transformers](https://www.sbert.net/) for text embedding. Any compatible model from HuggingFace works — just pass the model name.

Models are downloaded automatically from HuggingFace Hub to `~/.cache/huggingface/` on first use.

## Recommended Models

### Multilingual

| Model | Dimensions | Size | Languages | Description |
|-------|------------|------|-----------|-------------|
| `BAAI/bge-m3` | 1024 | ~2.2 GB | 100+ | **Default.** Best multilingual, dense+sparse |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | ~470 MB | 50+ | Fast multilingual |
| `intfloat/multilingual-e5-large` | 1024 | ~2.2 GB | 100+ | Strong multilingual |

### English

| Model | Dimensions | Size | Description |
|-------|------------|------|-------------|
| `BAAI/bge-small-en-v1.5` | 384 | ~130 MB | Tiny & fast, great for prototyping |
| `BAAI/bge-base-en-v1.5` | 768 | ~440 MB | Balanced quality/speed |
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3 GB | Highest quality English |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90 MB | Ultra lightweight |
| `thenlper/gte-small` | 384 | ~67 MB | Tiny but powerful |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~548 MB | Long context (8K tokens) |

## Using Models

```python
from DeepTextSearch import TextEmbedder

# Use the default model (BAAI/bge-m3)
embedder = TextEmbedder()

# Use any HuggingFace model
embedder = TextEmbedder("BAAI/bge-small-en-v1.5")
embedder = TextEmbedder("intfloat/e5-large-v2")
embedder = TextEmbedder("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
```

## Custom / Fine-tuned Models

You can use any model that is compatible with sentence-transformers:

```python
# Local fine-tuned model
embedder = TextEmbedder("./path/to/my-fine-tuned-model")

# Private HuggingFace model
embedder = TextEmbedder("your-org/your-private-model")
```

### Fine-tuning Your Own Model

Use the [sentence-transformers training API](https://www.sbert.net/docs/sentence_transformer/training_overview.html) to fine-tune on your domain data, then point DeepTextSearch to the saved directory:

```python
# After training and saving your model to ./my_model
embedder = TextEmbedder("./my_model")
embedder.index(your_corpus)
```

## GPU & CPU Device Selection

DeepTextSearch auto-detects the best available device (CUDA > MPS > CPU):

```python
# Auto-detect (default)
embedder = TextEmbedder()

# Force CPU
embedder = TextEmbedder(device="cpu")

# Force CUDA GPU
embedder = TextEmbedder(device="cuda")

# Force Apple Silicon GPU
embedder = TextEmbedder(device="mps")
```

## Encoding Without Indexing

Use `.encode()` to get raw embeddings without storing them:

```python
embedder = TextEmbedder()
vectors = embedder.encode(["Hello world", "AI is great"])
print(vectors.shape)  # (2, 1024) for bge-m3
```

## Batch Size

Control memory usage with the `batch_size` parameter:

```python
# Large batch for GPU (faster, more memory)
embedder = TextEmbedder(batch_size=256, device="cuda")

# Small batch for CPU (slower, less memory)
embedder = TextEmbedder(batch_size=16, device="cpu")
```

## Listing Available Presets

```python
from DeepTextSearch import EMBEDDING_PRESETS

for name, info in EMBEDDING_PRESETS.items():
    print(f"{name}: {info['description']} ({info['size_mb']} MB)")
```
