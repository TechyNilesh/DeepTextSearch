# Agentic AI Integration

## Overview

DeepTextSearch provides built-in integrations for modern AI agent frameworks:

| Integration | Install | Use Case |
|-------------|---------|----------|
| **MCP Server** | `pip install 'DeepTextSearch[mcp]'` | Claude Desktop, Claude Code, Cursor |
| **LangChain** | `pip install 'DeepTextSearch[langchain]'` | LangChain RAG chains and agents |
| **LlamaIndex** | `pip install 'DeepTextSearch[llamaindex]'` | LlamaIndex query engines |
| **Agent Tool** | Included | OpenAI/Claude function calling, custom agents |

## MCP Server

The built-in MCP server exposes search and reranking as tools that AI assistants can use directly.

### Setup

**1. Build your index:**
```python
from DeepTextSearch import TextEmbedder
embedder = TextEmbedder()
embedder.index(your_texts)
embedder.save("./my_index")
```

**2. Start the server:**
```bash
deep-text-search-mcp --index-path ./my_index
# With reranking
deep-text-search-mcp --index-path ./my_index --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2
# Force GPU
deep-text-search-mcp --index-path ./my_index --device cuda
```

**3. Configure Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "text-search": {
      "command": "deep-text-search-mcp",
      "args": ["--index-path", "/path/to/my_index"]
    }
  }
}
```

### MCP Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_texts` | `query`, `k=5`, `mode="hybrid"` | Search the indexed corpus |
| `rerank_passages` | `query`, `passages`, `top_n=5` | Rerank text passages |
| `get_index_info` | — | Get index metadata |

## LangChain Integration

Use DeepTextSearch as a drop-in retriever in any LangChain chain:

```python
from DeepTextSearch import TextEmbedder, Reranker
from DeepTextSearch.agents.langchain_retriever import create_langchain_retriever

embedder = TextEmbedder()
embedder.index(your_documents)

retriever = create_langchain_retriever(
    embedder,
    mode="hybrid",
    reranker=Reranker(),
    top_n=5,
)

# Use in a RAG chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n{context}\n\nQuestion: {question}"
)
llm = ChatOpenAI()

chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
answer = chain.invoke("What is machine learning?")
```

## LlamaIndex Integration

Use DeepTextSearch as a retriever in LlamaIndex query engines:

```python
from DeepTextSearch import TextEmbedder, Reranker
from DeepTextSearch.agents.llamaindex_retriever import create_llamaindex_retriever

embedder = TextEmbedder()
embedder.index(your_documents)

retriever = create_llamaindex_retriever(
    embedder,
    mode="hybrid",
    reranker=Reranker(),
    top_n=5,
)

from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(retriever)
response = query_engine.query("What is machine learning?")
```

## Generic Agent Tool

A callable tool that works with any AI agent framework:

```python
from DeepTextSearch import TextEmbedder, Reranker
from DeepTextSearch.agents import TextSearchTool

embedder = TextEmbedder()
embedder.index(your_texts)

tool = TextSearchTool(embedder, reranker=Reranker())

# Call it like a function — returns JSON
results = tool("what is machine learning?", k=5)

# Get function-calling schema (OpenAI/Claude compatible)
schema = tool.tool_definition
```

### Tool Definition Schema

The `.tool_definition` property returns a schema compatible with OpenAI and Claude function calling:

```json
{
  "type": "function",
  "function": {
    "name": "search_texts",
    "description": "Search a text corpus using semantic and keyword hybrid search.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "default": 5},
        "mode": {"type": "string", "enum": ["auto", "hybrid", "dense", "bm25"]}
      },
      "required": ["query"]
    }
  }
}
```
