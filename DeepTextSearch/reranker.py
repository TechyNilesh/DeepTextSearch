"""Cross-encoder reranker for search result refinement.

Retrieve broadly with TextSearch,
then rerank precisely with Reranker for maximum accuracy.
"""

from typing import Any, Dict, List, Optional, Union

from sentence_transformers import CrossEncoder

from .config import DEFAULT_RERANKER_MODEL, get_device


class RerankRequest:
    """Container for a rerank request.

    Args:
        query: The search query.
        passages: List of passage dicts, each must have a "text" key.
            Optionally include "id", "metadata", or any other fields.

    Example:
        >>> request = RerankRequest(
        ...     query="What is Python?",
        ...     passages=[
        ...         {"text": "Python is a programming language.", "id": 1},
        ...         {"text": "A python is a large snake.", "id": 2},
        ...     ]
        ... )
    """

    def __init__(self, query: str, passages: List[Dict[str, Any]]):
        self.query = query
        self.passages = passages


class Reranker:
    """Cross-encoder reranker for refining search results.

    Uses cross-encoder models that jointly encode query-passage pairs for
    much higher accuracy than bi-encoder similarity. Use this as a second
    stage after initial retrieval with TextSearch.

    Supports any cross-encoder model from HuggingFace, preset shorthands,
    or local/custom fine-tuned models.

    Args:
        model_name: HuggingFace model name, preset shorthand (e.g. "ms-marco-MiniLM-6"),
            or local path to a saved model.
        max_length: Maximum token length for query-passage pairs.
        device: Device for inference ('cpu', 'cuda', 'mps', or None for auto-detect).
        batch_size: Batch size for scoring.

    Example:
        >>> reranker = Reranker()

        # Use any HuggingFace model
        >>> reranker = Reranker("BAAI/bge-reranker-v2-m3")
        >>> reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-12-v2")

        # Use a local/custom model
        >>> reranker = Reranker("./my-fine-tuned-reranker")

        # Force GPU
        >>> reranker = Reranker(device="cuda")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        max_length: int = 512,
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = get_device(device)
        self._model = CrossEncoder(self.model_name, max_length=max_length, device=self.device)

    def rerank(
        self,
        request: RerankRequest,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank passages by relevance to the query.

        Args:
            request: A RerankRequest with query and passages.
            top_n: Return only the top N results. None returns all.

        Returns:
            List of passage dicts sorted by relevance, with "score" added.
        """
        if not request.passages:
            return []

        pairs = [[request.query, p["text"]] for p in request.passages]

        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        for passage, score in zip(request.passages, scores):
            passage["score"] = round(float(score), 6)

        ranked = sorted(request.passages, key=lambda p: p["score"], reverse=True)

        if top_n is not None:
            ranked = ranked[:top_n]

        return ranked

    def rerank_texts(
        self,
        query: str,
        texts: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience method: rerank a simple list of text strings.

        Args:
            query: The search query.
            texts: List of text strings to rerank.
            top_n: Return only the top N results.

        Returns:
            List of dicts with "text" and "score" keys, sorted by relevance.
        """
        passages = [{"text": t} for t in texts]
        request = RerankRequest(query=query, passages=passages)
        return self.rerank(request, top_n=top_n)

    def rerank_search_results(
        self,
        query: str,
        search_results: List,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank SearchResult objects from TextSearch.

        Args:
            query: The search query.
            search_results: List of SearchResult objects from TextSearch.search().
            top_n: Return only the top N results.

        Returns:
            List of dicts with "text", "score", "index", and "metadata" keys.
        """
        passages = [r.to_dict() for r in search_results]
        request = RerankRequest(query=query, passages=passages)
        return self.rerank(request, top_n=top_n)
