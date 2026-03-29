"""Demo 09: FAISS Index Types (flat, ivf, hnsw)

Shows the three FAISS index types and when to use each.
"""

import time
from DeepTextSearch import TextEmbedder, TextSearch

# Generate a larger corpus for benchmarking
corpus = [f"Document number {i} about topic {i % 10} with content about AI and search." for i in range(500)]

query = "AI search technology"

# Flat index — exact search, best accuracy
print("=== FAISS Flat (Exact) ===")
embedder_flat = TextEmbedder(
    model_name="BAAI/bge-small-en-v1.5",
    vector_store="faiss",
    index_type="flat",
)
embedder_flat.index(corpus)
search_flat = TextSearch(embedder_flat, mode="dense")
start = time.time()
results = search_flat.search(query, top_n=3)
flat_time = time.time() - start
print(f"  Time: {flat_time*1000:.1f}ms")
for r in results[:2]:
    print(f"  [{r.score:.4f}] {r.text[:60]}")

# HNSW index — fast approximate, good for large corpora
print("\n=== FAISS HNSW (Approximate) ===")
embedder_hnsw = TextEmbedder(
    model_name="BAAI/bge-small-en-v1.5",
    vector_store="faiss",
    index_type="hnsw",
)
embedder_hnsw.index(corpus)
search_hnsw = TextSearch(embedder_hnsw, mode="dense")
start = time.time()
results = search_hnsw.search(query, top_n=3)
hnsw_time = time.time() - start
print(f"  Time: {hnsw_time*1000:.1f}ms")
for r in results[:2]:
    print(f"  [{r.score:.4f}] {r.text[:60]}")

print(f"\n=== Summary ===")
print(f"  Flat:  {flat_time*1000:.1f}ms (exact)")
print(f"  HNSW:  {hnsw_time*1000:.1f}ms (approximate)")
