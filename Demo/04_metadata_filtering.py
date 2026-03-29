"""Demo 04: Metadata Storage and Filtering

Shows how to store metadata with documents and filter search results.
"""

import pandas as pd
from DeepTextSearch import TextEmbedder, TextSearch

# Create a DataFrame with text and metadata
data = {
    "content": [
        "Introduction to machine learning algorithms and their applications.",
        "Advanced deep learning with convolutional neural networks.",
        "Web development with React and modern JavaScript.",
        "Natural language processing with transformer models.",
        "Building REST APIs with Python Flask framework.",
        "Computer vision for autonomous driving systems.",
        "Data engineering pipelines with Apache Spark.",
        "Reinforcement learning for game playing agents.",
    ],
    "category": ["AI", "AI", "Web", "AI", "Web", "AI", "Data", "AI"],
    "level": ["beginner", "advanced", "intermediate", "advanced", "beginner", "advanced", "intermediate", "advanced"],
    "year": [2023, 2024, 2023, 2024, 2022, 2024, 2023, 2024],
}
df = pd.DataFrame(data)

# Index with metadata
embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder.index(df, text_column="content", metadata_columns=["category", "level", "year"])

search = TextSearch(embedder)

# Search without filter
print("=== All Results ===")
for r in search.search("neural networks", top_n=5):
    print(f"  [{r.score:.6f}] [{r.metadata}] {r.text}")

# Filter: only AI category
print("\n=== AI Category Only ===")
results = search.search(
    "neural networks",
    top_n=5,
    filter_fn=lambda text, meta: meta.get("category") == "AI",
)
for r in results:
    print(f"  [{r.score:.6f}] [{r.metadata['category']}] {r.text}")

# Filter: advanced level + year 2024
print("\n=== Advanced + Year 2024 ===")
results = search.search(
    "deep learning",
    top_n=5,
    filter_fn=lambda text, meta: meta.get("level") == "advanced" and meta.get("year") == 2024,
)
for r in results:
    print(f"  [{r.score:.6f}] [{r.metadata}] {r.text}")
