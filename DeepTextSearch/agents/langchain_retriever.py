"""LangChain retriever integration for DeepTextSearch.

Wraps TextSearch as a LangChain-compatible retriever for use in
RAG chains, agents, and LangChain pipelines.
"""

from typing import List, Optional

from ..embedder import TextEmbedder
from ..reranker import Reranker
from ..searcher import TextSearch


def create_langchain_retriever(
    embedder: TextEmbedder,
    mode: str = "hybrid",
    reranker: Optional[Reranker] = None,
    top_n: int = 5,
):
    """Create a LangChain-compatible retriever from a TextEmbedder.

    Args:
        embedder: TextEmbedder with indexed corpus.
        mode: Search mode ("hybrid", "dense", "bm25").
        reranker: Optional Reranker for result refinement.
        top_n: Number of results to return.

    Returns:
        A LangChain BaseRetriever instance.

    Example:
        >>> from DeepTextSearch import TextEmbedder
        >>> from DeepTextSearch.agents.langchain_retriever import create_langchain_retriever
        >>> embedder = TextEmbedder()
        >>> embedder.index(["doc1", "doc2", "doc3"])
        >>> retriever = create_langchain_retriever(embedder)
        >>> # Use in a LangChain chain
        >>> from langchain_core.runnables import RunnablePassthrough
        >>> chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    """
    try:
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    except ImportError:
        raise ImportError(
            "langchain-core is required for LangChain integration.\n"
            "Install with: pip install 'DeepTextSearch[langchain]'"
        )

    search = TextSearch(embedder, mode=mode)

    class DeepTextSearchRetriever(BaseRetriever):
        """LangChain retriever backed by DeepTextSearch."""

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        ) -> List[Document]:
            retrieve_n = top_n * 2 if reranker else top_n
            results = search.search(query, top_n=retrieve_n)

            if reranker and results:
                reranked = reranker.rerank_search_results(query, results, top_n=top_n)
                return [
                    Document(
                        page_content=r["text"],
                        metadata={**r.get("metadata", {}), "score": r["score"], "index": r.get("index")},
                    )
                    for r in reranked
                ]

            return [
                Document(
                    page_content=r.text,
                    metadata={**r.metadata, "score": r.score, "index": r.index},
                )
                for r in results[:top_n]
            ]

    return DeepTextSearchRetriever()
