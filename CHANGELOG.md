# Changelog

All notable changes to DeepTextSearch will be documented in this file.

---

## [1.0.0] — 2026-03-29

**Complete rewrite.** DeepTextSearch has been rebuilt from the ground up for the modern AI era.

### Added

- **Hybrid Search** — dense semantic search + BM25 keyword search combined with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — `Reranker` class with support for tiny (~17 MB) to large (~2.2 GB) cross-encoder models
- **Pluggable Vector Store Backends**
  - `FAISSStore` — local FAISS index with flat/ivf/hnsw support
  - `ChromaStore` — ChromaDB persistent and in-memory backend
  - `QdrantStore` — Qdrant local, in-memory, and remote server
  - `PostgresStore` — PostgreSQL + pgvector with custom metadata schema
  - `MongoStore` — MongoDB Atlas Vector Search with top-level metadata fields
  - `BaseVectorStore` — abstract interface for custom backends
- **`store_config` parameter** — pass connection strings, table names, metadata schemas, and other backend-specific settings via dict
- **Pluggable Embedding Models** — use any sentence-transformers model from HuggingFace or local fine-tuned models
- **GPU/CPU Auto-Detection** — automatic CUDA > MPS > CPU device selection with explicit override
- **MCP Server** — built-in Model Context Protocol server (`deep-text-search-mcp` CLI) for Claude Desktop, Claude Code, Cursor
- **LangChain Integration** — `create_langchain_retriever()` for drop-in RAG chain support
- **LlamaIndex Integration** — `create_llamaindex_retriever()` for query engine support
- **Agent Tool Interface** — `TextSearchTool` callable with OpenAI/Claude function-calling schema
- **Incremental Indexing** — `.add()` and `.delete()` methods for dynamic index updates
- **Metadata Support** — store, persist, and filter by document metadata
- **DataFrame/CSV Support** — `.index()` accepts list, Series, or DataFrame; `.from_csv()` for one-line loading
- **Persistent Storage** — save/load indexes to any local folder path (FAISS + JSON)
- **`SearchResult` class** — structured result objects with `.to_dict()` method
- **`RerankRequest` class** — structured reranking input container
- **9 embedding model presets** with dimensions, size, and language info
- **9 reranker model presets** including 3 tiny models (17–45 MB) for CPU/edge deployments
- **CITATION.cff** — GitHub-native citation support
- **Full documentation** — 8 detailed doc files in `Documents/` folder

### Changed

- **Default embedding model** — `paraphrase-xlm-r-multilingual-v1` → `BAAI/bge-m3` (100+ languages, 1024 dims)
- **Search method** — brute-force cosine similarity → FAISS ANN indexing (10–1000x faster at scale)
- **Storage format** — pickle files → FAISS index + JSON (portable, human-readable)
- **API design** — removed all `input()` calls; fully programmatic, production-ready API
- **Packaging** — `setup.py` + `setup.cfg` → modern `pyproject.toml` with optional dependency groups
- **Python support** — 3.4–3.9 → 3.9+

### Removed

- `LoadData` class — replaced by `TextEmbedder.from_csv()` and direct DataFrame/list support
- `input()` prompts for column selection and re-embedding confirmation
- `embedding-data/` hardcoded folder — replaced by configurable `index_dir`
- Pickle-based storage — replaced by FAISS + JSON
- `setup.py`, `setup.cfg`, `requirements.txt` — replaced by `pyproject.toml`
- Old `DeepTextSearch.py` single-file architecture

---

## [0.3] — 2021-05-31

### Changed

- Updated `setup.py` with new version and metadata
- Updated README with improved documentation

---

## [0.2] — 2021-05-30

### Added

- `TextEmbedder` class with `paraphrase-xlm-r-multilingual-v1` sentence-transformer model
- `TextSearch` class with cosine similarity-based search
- `LoadData` class for CSV and text file loading
- Pickle-based embedding persistence in `embedding-data/` folder
- Demo script and Jupyter notebook
- Multilingual support (50+ languages)

---

## [0.1] — 2021-05-30

### Added

- Initial release
- Basic text search using sentence-transformers
- CSV data loading with interactive column selection
