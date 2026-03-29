"""Demo 06: Incremental Indexing

Shows how to add and delete documents without re-embedding everything.
"""

from DeepTextSearch import TextEmbedder, TextSearch

# Start with initial corpus
corpus = [
    "Python is great for data science.",
    "JavaScript powers the modern web.",
    "Rust provides memory safety without garbage collection.",
]

embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder.index(corpus)
print(f"Initial corpus size: {embedder.corpus_size}")

# Add a single document
embedder.add("Go is designed for concurrent programming.")
print(f"After adding 1 doc: {embedder.corpus_size}")

# Add multiple documents with metadata
embedder.add(
    ["TypeScript adds type safety to JavaScript.", "C++ is used for system programming."],
    metadata=[{"category": "web"}, {"category": "systems"}],
)
print(f"After adding 2 more: {embedder.corpus_size}")

# Search the updated index
search = TextSearch(embedder)
print("\n=== Search Updated Index ===")
for r in search.search("type safety in programming", top_n=3):
    print(f"  [{r.score:.4f}] {r.text}")

# Delete documents
embedder.delete([0, 1])  # Remove first two documents
print(f"\nAfter deleting 2 docs: {embedder.corpus_size}")
