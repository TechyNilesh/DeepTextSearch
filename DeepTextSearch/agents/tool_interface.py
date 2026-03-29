"""Generic tool interface for agentic AI frameworks.

Provides a callable tool that any AI agent framework can use
(OpenAI function calling, Claude tool use, custom agents, etc.).
"""

import json
from typing import Any, Dict, List, Optional

from ..embedder import TextEmbedder
from ..searcher import TextSearch
from ..reranker import Reranker


class TextSearchTool:
    """A callable search tool for AI agents.

    Wraps TextSearch + optional Reranker into a single callable
    that agents can invoke. Also exposes a tool_definition property
    compatible with OpenAI/Claude function-calling schemas.

    Args:
        embedder: TextEmbedder with indexed corpus.
        mode: Search mode ("hybrid", "dense", "bm25").
        reranker: Optional Reranker for result refinement.

    Example:
        >>> embedder = TextEmbedder()
        >>> embedder.index(["Python is great", "Java is popular"])
        >>> tool = TextSearchTool(embedder)
        >>> result = tool("best programming language", k=3)
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        mode: str = "hybrid",
        reranker: Optional[Reranker] = None,
    ):
        self._search = TextSearch(embedder, mode=mode)
        self._reranker = reranker

    def __call__(self, query: str, k: int = 5, mode: str = "auto") -> str:
        """Search the indexed corpus and return JSON results.

        Args:
            query: Search query text.
            k: Number of results to return.
            mode: Search mode ("hybrid", "dense", "bm25", or "auto" for default).

        Returns:
            JSON string of search results.
        """
        search_mode = None if mode == "auto" else mode
        results = self._search.search(query, top_n=k * 2 if self._reranker else k, mode=search_mode)

        if self._reranker and results:
            reranked = self._reranker.rerank_search_results(query, results, top_n=k)
            return json.dumps(reranked, indent=2, ensure_ascii=False)

        return json.dumps(
            [r.to_dict() for r in results[:k]],
            indent=2,
            ensure_ascii=False,
        )

    @property
    def tool_definition(self) -> Dict[str, Any]:
        """OpenAI/Claude function-calling compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "search_texts",
                "description": "Search a text corpus using semantic and keyword hybrid search. Returns the most relevant text passages for a given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query text.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return.",
                            "default": 5,
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "hybrid", "dense", "bm25"],
                            "description": "Search mode. 'auto' uses the default configured mode.",
                            "default": "auto",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
