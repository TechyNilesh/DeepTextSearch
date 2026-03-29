"""LlamaIndex retriever integration for DeepTextSearch.

Wraps TextSearch as a LlamaIndex-compatible retriever for use in
RAG pipelines, query engines, and LlamaIndex agents.
"""

from typing import List, Optional

from ..embedder import TextEmbedder
from ..reranker import Reranker
from ..searcher import TextSearch


def create_llamaindex_retriever(
    embedder: TextEmbedder,
    mode: str = "hybrid",
    reranker: Optional[Reranker] = None,
    top_n: int = 5,
):
    """Create a LlamaIndex-compatible retriever from a TextEmbedder.

    Args:
        embedder: TextEmbedder with indexed corpus.
        mode: Search mode ("hybrid", "dense", "bm25").
        reranker: Optional Reranker for result refinement.
        top_n: Number of results to return.

    Returns:
        A LlamaIndex BaseRetriever instance.

    Example:
        >>> from DeepTextSearch import TextEmbedder
        >>> from DeepTextSearch.agents.llamaindex_retriever import create_llamaindex_retriever
        >>> embedder = TextEmbedder()
        >>> embedder.index(["doc1", "doc2", "doc3"])
        >>> retriever = create_llamaindex_retriever(embedder)
        >>> # Use in a LlamaIndex query engine
        >>> from llama_index.core.query_engine import RetrieverQueryEngine
        >>> query_engine = RetrieverQueryEngine.from_args(retriever)
    """
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    except ImportError:
        raise ImportError(
            "llama-index-core is required for LlamaIndex integration.\n"
            "Install with: pip install 'DeepTextSearch[llamaindex]'"
        )

    search = TextSearch(embedder, mode=mode)

    class DeepTextSearchRetriever(BaseRetriever):
        """LlamaIndex retriever backed by DeepTextSearch."""

        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            query = query_bundle.query_str
            retrieve_n = top_n * 2 if reranker else top_n
            results = search.search(query, top_n=retrieve_n)

            if reranker and results:
                reranked = reranker.rerank_search_results(query, results, top_n=top_n)
                return [
                    NodeWithScore(
                        node=TextNode(
                            text=r["text"],
                            metadata={**r.get("metadata", {}), "index": r.get("index")},
                        ),
                        score=r["score"],
                    )
                    for r in reranked
                ]

            return [
                NodeWithScore(
                    node=TextNode(
                        text=r.text,
                        metadata={**r.metadata, "index": r.index},
                    ),
                    score=r.score,
                )
                for r in results[:top_n]
            ]

    return DeepTextSearchRetriever()
