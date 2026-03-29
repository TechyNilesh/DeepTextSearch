"""Demo 02: Hybrid, Dense, and BM25 Search Modes

Compares the three search modes on the same query.
"""

from DeepTextSearch import TextEmbedder, TextSearch

corpus = [
    "The Python programming language was created by Guido van Rossum.",
    "Python snakes are found in tropical regions of Africa and Asia.",
    "PyTorch is a deep learning framework written in Python.",
    "Django is a Python web framework for rapid development.",
    "Anaconda is a Python distribution for data science.",
    "The reticulated python is the longest snake in the world.",
    "Flask is a lightweight Python web microframework.",
    "Python 3.12 introduced significant performance improvements.",
]

embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder.index(corpus)
search = TextSearch(embedder)

query = "Python programming language"

# Hybrid search (default) — combines semantic + keyword
print("=== Hybrid Search ===")
for r in search.search(query, top_n=3, mode="hybrid"):
    print(f"  [{r.score:.6f}] {r.text}")

# Dense search — pure semantic similarity
print("\n=== Dense Search ===")
for r in search.search(query, top_n=3, mode="dense"):
    print(f"  [{r.score:.4f}] {r.text}")

# BM25 search — pure keyword matching
print("\n=== BM25 Search ===")
for r in search.search(query, top_n=3, mode="bm25"):
    print(f"  [{r.score:.4f}] {r.text}")

# Tune hybrid weights
print("\n=== Hybrid (keyword-heavy: bm25=0.7, dense=0.3) ===")
search_kw = TextSearch(embedder, dense_weight=0.3, bm25_weight=0.7)
for r in search_kw.search(query, top_n=3):
    print(f"  [{r.score:.6f}] {r.text}")
