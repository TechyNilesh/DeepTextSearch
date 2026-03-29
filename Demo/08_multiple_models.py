"""Demo 08: Using Different Embedding and Reranker Models

Shows how to use different models for different use cases.
"""

from DeepTextSearch import TextEmbedder, TextSearch, Reranker, EMBEDDING_PRESETS, RERANKER_PRESETS

corpus = [
    "Artificial intelligence is transforming healthcare diagnostics.",
    "Deep reinforcement learning achieved superhuman performance in games.",
    "Natural language understanding enables conversational AI assistants.",
    "Computer vision models can detect objects in real-time video.",
    "Generative AI creates realistic images, text, and audio content.",
]

# List available presets
print("=== Available Embedding Presets ===")
for name, info in EMBEDDING_PRESETS.items():
    print(f"  {name}: {info['dimensions']}d, ~{info['size_mb']} MB, {info['languages']}")

print("\n=== Available Reranker Presets ===")
for name, info in RERANKER_PRESETS.items():
    print(f"  {name}: ~{info['size_mb']} MB, {info['languages']}")

# Use a tiny model for fast prototyping
print("\n=== Tiny Model (bge-small-en-v1.5, ~130 MB) ===")
embedder_small = TextEmbedder("BAAI/bge-small-en-v1.5")
embedder_small.index(corpus)
search_small = TextSearch(embedder_small)
for r in search_small.search("AI in healthcare", top_n=2):
    print(f"  [{r.score:.4f}] {r.text}")

# Use a tiny reranker (~17 MB) for CPU/edge
print("\n=== Tiny Reranker (TinyBERT, ~17 MB) ===")
reranker_tiny = Reranker("cross-encoder/ms-marco-TinyBERT-L-2-v2")
results = search_small.search("generative AI", top_n=5)
reranked = reranker_tiny.rerank_search_results("generative AI", results, top_n=3)
for r in reranked:
    print(f"  [{r['score']:.4f}] {r['text']}")

# Check device
print(f"\nEmbedder device: {embedder_small.device}")
print(f"Reranker device: {reranker_tiny.device}")
