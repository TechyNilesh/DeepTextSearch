"""Demo 11: Agent Tool Interface

Shows how to use DeepTextSearch as a callable tool for AI agents.
"""

import json
from DeepTextSearch import TextEmbedder, Reranker
from DeepTextSearch.agents import TextSearchTool

corpus = [
    "Python is widely used in data science and machine learning.",
    "JavaScript powers interactive web applications.",
    "Rust ensures memory safety without a garbage collector.",
    "Go is designed for building scalable network services.",
    "TypeScript adds static typing to JavaScript.",
]

# Create embedder and index
embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
embedder.index(corpus)

# Create tool without reranker
tool = TextSearchTool(embedder)

# Call it like a function — returns JSON
print("=== Tool Call (no reranker) ===")
result = tool("best language for web development", k=3)
parsed = json.loads(result)
for r in parsed:
    print(f"  [{r['score']:.4f}] {r['text']}")

# Create tool with reranker
tool_with_reranker = TextSearchTool(
    embedder,
    reranker=Reranker("cross-encoder/ms-marco-TinyBERT-L-2-v2"),
)

print("\n=== Tool Call (with reranker) ===")
result = tool_with_reranker("memory safe programming", k=3)
parsed = json.loads(result)
for r in parsed:
    print(f"  [{r['score']:.4f}] {r['text']}")

# Get the function-calling schema (OpenAI/Claude compatible)
print("\n=== Tool Definition (for function calling) ===")
print(json.dumps(tool.tool_definition, indent=2))
