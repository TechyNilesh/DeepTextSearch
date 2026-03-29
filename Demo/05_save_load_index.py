"""Demo 05: Save and Load Index

Shows how to persist an index to disk and reload it later.
"""

import tempfile
import os
from DeepTextSearch import TextEmbedder, TextSearch

corpus = [
    "Quantum computing uses qubits for parallel computation.",
    "Classical computers use binary bits for processing.",
    "Machine learning models learn patterns from data.",
    "Cloud computing provides on-demand computing resources.",
    "Edge computing processes data closer to the source.",
]

# Create and save index
embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder.index(corpus)

save_path = os.path.join(tempfile.gettempdir(), "demo_index")
embedder.save(save_path)
print(f"Index saved to: {save_path}")
print(f"Saved files: {os.listdir(save_path)}")

# Load index in a new session (no re-embedding needed)
loaded_embedder = TextEmbedder.load(save_path)
print(f"\nLoaded index: {loaded_embedder.corpus_size} documents, {loaded_embedder.dimension}d embeddings")

# Search on loaded index
search = TextSearch(loaded_embedder)
results = search.search("parallel processing", top_n=3)

print("\n=== Search on Loaded Index ===")
for r in results:
    print(f"  [{r.score:.4f}] {r.text}")
