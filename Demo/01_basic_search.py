"""Demo 01: Basic Text Search

Shows the simplest usage — index a list of texts and search.
"""

from DeepTextSearch import TextEmbedder, TextSearch

# Sample corpus
corpus = [
    "Python is a versatile programming language used in web development, data science, and AI.",
    "JavaScript is the most popular language for frontend web development.",
    "Machine learning is a subset of artificial intelligence focused on learning from data.",
    "Deep learning uses neural networks with many layers to model complex patterns.",
    "Natural language processing enables computers to understand human language.",
    "FAISS is a library for efficient similarity search developed by Meta.",
    "Transfer learning allows models to leverage knowledge from pre-trained networks.",
    "Transformers architecture revolutionized NLP with the attention mechanism.",
    "Kubernetes orchestrates containerized applications at scale.",
    "PostgreSQL is a powerful open-source relational database.",
]

# Index
embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder.index(corpus)

# Search
search = TextSearch(embedder)
results = search.search("What is deep learning?", top_n=5)

print("=== Basic Search Results ===")
for r in results:
    print(f"  [{r.score:.4f}] {r.text}")
