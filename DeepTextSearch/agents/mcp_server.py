"""MCP (Model Context Protocol) server for DeepTextSearch.

Exposes text search and reranking as MCP tools that AI agents
(Claude Desktop, Claude Code, Cursor, etc.) can use directly.

Usage:
    # Start the MCP server
    deep-text-search-mcp --index-path ./my_index

    # Or run directly
    python -m DeepTextSearch.agents.mcp_server --index-path ./my_index
"""

import argparse
import json
import sys
from typing import Optional


def create_mcp_server(
    index_path: str,
    model_name: Optional[str] = None,
    reranker_model: Optional[str] = None,
    device: Optional[str] = None,
):
    """Create and configure an MCP server with search tools.

    Args:
        index_path: Path to saved TextEmbedder index directory.
        model_name: Override embedding model (uses saved config by default).
        reranker_model: Reranker model name (None to disable reranking).
        device: Device for inference.

    Returns:
        Configured FastMCP server instance.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print(
            "MCP support requires the 'mcp' package.\n"
            "Install with: pip install 'DeepTextSearch[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from ..embedder import TextEmbedder
    from ..searcher import TextSearch
    from ..reranker import Reranker

    # Load index
    embedder = TextEmbedder.load(index_path, device=device)
    search = TextSearch(embedder, mode="hybrid")
    reranker = Reranker(model_name=reranker_model, device=device) if reranker_model else None

    mcp = FastMCP("DeepTextSearch")

    @mcp.tool()
    def search_texts(query: str, k: int = 5, mode: str = "hybrid") -> str:
        """Search the indexed text corpus using semantic and keyword hybrid search.

        Args:
            query: The search query text.
            k: Number of results to return (default 5).
            mode: Search mode - "hybrid" (default), "dense" (semantic only), or "bm25" (keyword only).

        Returns:
            JSON array of results with text, score, index, and metadata.
        """
        results = search.search(query, top_n=k * 2 if reranker else k, mode=mode)

        if reranker and results:
            reranked = reranker.rerank_search_results(query, results, top_n=k)
            return json.dumps(reranked, indent=2, ensure_ascii=False)

        return json.dumps(
            [r.to_dict() for r in results[:k]],
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def rerank_passages(query: str, passages: list[str], top_n: int = 5) -> str:
        """Rerank a list of text passages by relevance to a query using a cross-encoder model.

        Args:
            query: The search query.
            passages: List of text passages to rerank.
            top_n: Number of top results to return.

        Returns:
            JSON array of reranked passages with scores.
        """
        if reranker is None:
            return json.dumps({"error": "Reranker not configured. Start server with --reranker-model."})

        ranked = reranker.rerank_texts(query, passages, top_n=top_n)
        return json.dumps(ranked, indent=2, ensure_ascii=False)

    @mcp.tool()
    def get_index_info() -> str:
        """Get information about the loaded text search index.

        Returns:
            JSON with corpus_size, model, dimension, device, and search mode.
        """
        info = {
            "corpus_size": embedder.corpus_size,
            "model": embedder.model_name,
            "dimension": embedder.dimension,
            "device": embedder.device,
            "search_mode": "hybrid",
            "reranker": reranker.model_name if reranker else None,
        }
        return json.dumps(info, indent=2)

    return mcp


def main():
    """CLI entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="DeepTextSearch MCP Server")
    parser.add_argument(
        "--index-path",
        required=True,
        help="Path to saved TextEmbedder index directory.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override embedding model name.",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Reranker model name. Omit to disable reranking.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device for inference (auto-detected if not set).",
    )
    args = parser.parse_args()

    mcp = create_mcp_server(
        index_path=args.index_path,
        model_name=args.model,
        reranker_model=args.reranker_model,
        device=args.device,
    )
    mcp.run()


if __name__ == "__main__":
    main()
