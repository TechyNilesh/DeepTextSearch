"""Demo 10: Complete RAG (Retrieval Augmented Generation) Pipeline

Shows a full retrieve → rerank → generate pipeline.
The LLM call is shown as a placeholder — replace with your preferred LLM.
"""

from DeepTextSearch import TextEmbedder, TextSearch, Reranker

# Knowledge base
knowledge_base = [
    "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It supports billion-scale datasets.",
    "ChromaDB is an open-source embedding database designed for AI applications. It provides simple APIs for storing and querying embeddings.",
    "Qdrant is a vector similarity search engine with extended filtering support. It is written in Rust for performance.",
    "PostgreSQL with pgvector extension enables vector similarity search directly in your relational database.",
    "MongoDB Atlas Vector Search allows you to perform semantic search on your MongoDB data using vector embeddings.",
    "Hybrid search combines keyword-based (BM25) and semantic (dense vector) search using techniques like Reciprocal Rank Fusion.",
    "Cross-encoder rerankers jointly encode query-passage pairs for much higher accuracy than bi-encoder similarity, but are slower.",
    "Sentence transformers are pre-trained models that convert text into dense vector embeddings capturing semantic meaning.",
    "BM25 is a probabilistic retrieval function that ranks documents based on term frequency and inverse document frequency.",
    "Reciprocal Rank Fusion (RRF) combines results from multiple retrieval systems by aggregating their reciprocal ranks.",
]

# Step 1: Index the knowledge base
print("Step 1: Indexing knowledge base...")
embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder.index(knowledge_base)

# Step 2: Hybrid search to retrieve candidates
print("Step 2: Retrieving candidates...")
search = TextSearch(embedder, mode="hybrid")
query = "How does hybrid search work and why is it better than pure semantic search?"
candidates = search.search(query, top_n=10)

print(f"\n=== Retrieved {len(candidates)} Candidates ===")
for i, r in enumerate(candidates):
    print(f"  {i+1}. [{r.score:.6f}] {r.text[:80]}...")

# Step 3: Rerank for precision
print("\nStep 3: Reranking...")
reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank_search_results(query, candidates, top_n=3)

print(f"\n=== Top 3 After Reranking ===")
for i, r in enumerate(reranked):
    print(f"  {i+1}. [{r['score']:.4f}] {r['text'][:80]}...")

# Step 4: Build context for LLM
context = "\n".join([f"- {doc['text']}" for doc in reranked])

prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

print(f"\n=== Prompt for LLM ===")
print(prompt)
print("\n(Replace this with an actual LLM call: OpenAI, Anthropic, local model, etc.)")
