"""Demo 03: Cross-Encoder Reranking

Shows how to retrieve broadly and then rerank for precision.
"""

from DeepTextSearch import TextEmbedder, TextSearch, Reranker, RerankRequest

corpus = [
    "FAISS enables fast approximate nearest neighbor search on billions of vectors.",
    "PostgreSQL supports full-text search with GIN indexes and tsvector.",
    "Elasticsearch is a distributed search engine built on Apache Lucene.",
    "Redis can be used as an in-memory cache for search results.",
    "Kubernetes manages container orchestration across clusters.",
    "ChromaDB is an open-source vector database for AI applications.",
    "Qdrant is a vector similarity search engine with filtering support.",
    "MongoDB Atlas offers vector search capabilities for AI workloads.",
    "Apache Solr provides enterprise search with faceted navigation.",
    "Milvus is designed for billion-scale vector similarity search.",
]

embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder.index(corpus)
search = TextSearch(embedder)

query = "fastest vector database for similarity search"

# Step 1: Retrieve candidates
results = search.search(query, top_n=8)
print("=== Before Reranking ===")
for r in results:
    print(f"  [{r.score:.6f}] {r.text}")

# Step 2: Rerank with cross-encoder
reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank_search_results(query, results, top_n=5)

print("\n=== After Reranking ===")
for r in reranked:
    print(f"  [{r['score']:.4f}] {r['text']}")

# Standalone reranking (without prior search)
print("\n=== Standalone Reranking ===")
request = RerankRequest(
    query="efficient vector search",
    passages=[
        {"text": "FAISS enables billion-scale similarity search.", "id": 1},
        {"text": "PostgreSQL supports full-text search.", "id": 2},
        {"text": "Kubernetes manages containers.", "id": 3},
    ],
)
ranked = reranker.rerank(request)
for r in ranked:
    print(f"  [{r['score']:.4f}] (id={r['id']}) {r['text']}")

# Rerank plain text strings
print("\n=== Rerank Plain Texts ===")
ranked = reranker.rerank_texts(
    "machine learning framework",
    ["PyTorch is a deep learning framework.", "Django is a web framework.", "TensorFlow is for ML."],
)
for r in ranked:
    print(f"  [{r['score']:.4f}] {r['text']}")
