"""Demo 07: Loading Data from CSV and DataFrame

Shows different ways to load and index text data.
"""

import tempfile
import os
import pandas as pd
from DeepTextSearch import TextEmbedder, TextSearch

# Create a sample CSV for demonstration
csv_path = os.path.join(tempfile.gettempdir(), "sample_articles.csv")
df = pd.DataFrame({
    "title": [
        "Intro to ML",
        "Web Dev Guide",
        "NLP Tutorial",
        "Database Design",
        "Cloud Architecture",
    ],
    "content": [
        "Machine learning algorithms learn patterns from training data to make predictions.",
        "Modern web development uses React, Vue, or Angular for building user interfaces.",
        "Natural language processing transforms how computers understand human text.",
        "Relational database design focuses on normalization and efficient query patterns.",
        "Cloud architecture principles include scalability, reliability, and cost optimization.",
    ],
    "category": ["AI", "Web", "AI", "Data", "Cloud"],
    "author": ["Alice", "Bob", "Alice", "Charlie", "Diana"],
})
df.to_csv(csv_path, index=False)

# Method 1: From CSV (one-liner)
print("=== Method 1: From CSV ===")
embedder = TextEmbedder.from_csv(
    csv_path,
    text_column="content",
    metadata_columns=["title", "category", "author"],
    model_name="BAAI/bge-small-en-v1.5",
)
search = TextSearch(embedder)
for r in search.search("machine learning", top_n=2):
    print(f"  [{r.score:.4f}] [{r.metadata.get('title')}] {r.text}")

# Method 2: From DataFrame
print("\n=== Method 2: From DataFrame ===")
embedder2 = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder2.index(df, text_column="content", metadata_columns=["title", "category"])
search2 = TextSearch(embedder2)
for r in search2.search("database", top_n=2):
    print(f"  [{r.score:.4f}] [{r.metadata.get('category')}] {r.text}")

# Method 3: From Series
print("\n=== Method 3: From Series ===")
embedder3 = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder3.index(df["content"])
search3 = TextSearch(embedder3)
for r in search3.search("web development", top_n=2):
    print(f"  [{r.score:.4f}] {r.text}")

# Method 4: From plain list
print("\n=== Method 4: From List ===")
embedder4 = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder4.index(["Hello world", "AI is amazing", "Python rocks"])
search4 = TextSearch(embedder4)
for r in search4.search("artificial intelligence", top_n=2):
    print(f"  [{r.score:.4f}] {r.text}")

# Cleanup
os.remove(csv_path)
