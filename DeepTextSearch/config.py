"""Default configurations and model presets for DeepTextSearch."""

import torch

# ──────────────────────────────────────────────
# Embedding Model Presets
# ──────────────────────────────────────────────
# These are recommended models. Any sentence-transformers compatible model
# from HuggingFace works — just pass the model name/path.

EMBEDDING_PRESETS = {
    # Multilingual
    "BAAI/bge-m3": {
        "dimensions": 1024,
        "max_tokens": 8192,
        "size_mb": 2200,
        "languages": "100+",
        "description": "Best multilingual. Highest quality, dense+sparse support.",
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "dimensions": 384,
        "max_tokens": 512,
        "size_mb": 470,
        "languages": "50+",
        "description": "Fast multilingual. Good quality/speed trade-off.",
    },
    "intfloat/multilingual-e5-large": {
        "dimensions": 1024,
        "max_tokens": 512,
        "size_mb": 2240,
        "languages": "100+",
        "description": "Strong multilingual from Microsoft. High quality.",
    },
    # English
    "BAAI/bge-small-en-v1.5": {
        "dimensions": 384,
        "max_tokens": 512,
        "size_mb": 130,
        "languages": "English",
        "description": "Tiny & fast. Great for prototyping.",
    },
    "BAAI/bge-base-en-v1.5": {
        "dimensions": 768,
        "max_tokens": 512,
        "size_mb": 440,
        "languages": "English",
        "description": "Balanced quality and speed.",
    },
    "BAAI/bge-large-en-v1.5": {
        "dimensions": 1024,
        "max_tokens": 512,
        "size_mb": 1340,
        "languages": "English",
        "description": "Highest quality English model.",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimensions": 384,
        "max_tokens": 512,
        "size_mb": 90,
        "languages": "English",
        "description": "Ultra lightweight. Fastest inference.",
    },
    "thenlper/gte-small": {
        "dimensions": 384,
        "max_tokens": 512,
        "size_mb": 67,
        "languages": "English",
        "description": "Alibaba GTE. Tiny but powerful.",
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "dimensions": 768,
        "max_tokens": 8192,
        "size_mb": 548,
        "languages": "English",
        "description": "Long context (8K tokens). Open source.",
    },
}

# ──────────────────────────────────────────────
# Reranker Model Presets
# ──────────────────────────────────────────────

RERANKER_PRESETS = {
    # Tiny models (< 100 MB) — great for CPU / serverless / edge
    "cross-encoder/ms-marco-TinyBERT-L-2-v2": {
        "max_tokens": 512,
        "size_mb": 17,
        "languages": "English",
        "description": "Smallest reranker (~17 MB). Ultra fast, CPU-friendly.",
    },
    "cross-encoder/ms-marco-MiniLM-L-2-v2": {
        "max_tokens": 512,
        "size_mb": 17,
        "languages": "English",
        "description": "Tiny 2-layer reranker. Minimal latency.",
    },
    "cross-encoder/ms-marco-MiniLM-L-4-v2": {
        "max_tokens": 512,
        "size_mb": 45,
        "languages": "English",
        "description": "Small 4-layer reranker. Good for resource-constrained environments.",
    },
    # Standard models
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "max_tokens": 512,
        "size_mb": 90,
        "languages": "English",
        "description": "Best speed/quality ratio. Recommended default.",
    },
    "cross-encoder/ms-marco-MiniLM-L-12-v2": {
        "max_tokens": 512,
        "size_mb": 130,
        "languages": "English",
        "description": "Higher quality English reranker.",
    },
    "BAAI/bge-reranker-v2-m3": {
        "max_tokens": 8192,
        "size_mb": 2200,
        "languages": "100+",
        "description": "Best multilingual reranker.",
    },
    "BAAI/bge-reranker-base": {
        "max_tokens": 512,
        "size_mb": 1110,
        "languages": "English",
        "description": "Balanced English reranker.",
    },
    "BAAI/bge-reranker-large": {
        "max_tokens": 512,
        "size_mb": 1340,
        "languages": "English",
        "description": "Highest accuracy reranker.",
    },
    "jinaai/jina-reranker-v2-base-multilingual": {
        "max_tokens": 1024,
        "size_mb": 1110,
        "languages": "100+",
        "description": "Jina multilingual reranker.",
    },
}

# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_INDEX_DIR = ".deeptextsearch"


def get_device(device: str = None) -> str:
    """Auto-detect the best available device.

    Args:
        device: Explicit device ('cpu', 'cuda', 'mps') or None for auto-detect.

    Returns:
        Device string.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
