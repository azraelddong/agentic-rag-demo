from abc import ABC, abstractmethod

from app.rag.vector_store import SearchResult


class Reranker(ABC):
    """Rerank retrieved chunks before prompt construction."""

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return reranked search results."""


class NoopReranker(Reranker):
    """Default reranker for phase 1. Keeps Milvus ranking unchanged."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results


class BgeReranker(Reranker):
    """Reserved extension point for bge-reranker."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        raise NotImplementedError("bge-reranker 预留：可在此接入本地模型或 rerank 服务")
