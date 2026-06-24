from abc import ABC, abstractmethod
import logging

import httpx

from app.core.exceptions import ConfigurationError, LLMProviderError
from app.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Rerank retrieved chunks before prompt construction."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """Return reranked search results, optionally trimmed to top_n."""


class NoopReranker(Reranker):
    """Default reranker for phase 1. Keeps Milvus ranking unchanged."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        return results


class APIReranker(Reranker):
    """Base class for HTTP-based rerank APIs (OpenAI-compatible /v1/rerank)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(30.0))
        return self._client

    def _check_api_key(self) -> None:
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ConfigurationError(
                "Rerank API Key 未配置，请先在 .env 中填写 RERANK_API_KEY 或 LLM_API_KEY"
            )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        if not results:
            return []

        self._check_api_key()
        documents = [result.text for result in results]

        try:
            response = self.client.post(
                f"{self.base_url}/rerank",
                json=self._build_payload(query, documents, top_n),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            logger.exception("Rerank API returned an error status")
            raise LLMProviderError(
                "Rerank API 调用失败",
                detail={"status": exc.response.status_code, "error": str(exc)},
            ) from exc
        except httpx.RequestError as exc:
            logger.exception("Rerank API request failed")
            raise LLMProviderError(
                "Rerank API 请求失败",
                detail={"error": str(exc)},
            ) from exc

        return self._apply_rerank_results(body, results, top_n)

    def _build_payload(
        self,
        query: str,
        documents: list[str],
        top_n: int | None,
    ) -> dict:
        payload: dict = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        return payload

    def _apply_rerank_results(
        self,
        body: dict,
        results: list[SearchResult],
        top_n: int | None,
    ) -> list[SearchResult]:
        """Reorder results by API-returned relevance_score order, update scores, and trim."""
        rerank_items = body.get("results", [])
        if not rerank_items:
            logger.warning("Rerank API returned no results, falling back to original order")
            return results[:top_n] if top_n else results

        # Build old-score lookup for change logging
        old_scores = {i: results[i].score for i in range(len(results))}

        reordered: list[SearchResult] = []
        for item in rerank_items:
            idx = item.get("index")
            new_score = item.get("relevance_score")
            if isinstance(idx, int) and 0 <= idx < len(results):
                result = results[idx]
                old_score = old_scores.get(idx, 0.0)
                result.score = float(new_score)
                reordered.append(result)
                logger.info(
                    "Rerank score: file=%s chunk=%s  old=%.4f → new=%.4f  (Δ=%+.4f)",
                    result.metadata.get("file_name", "?"),
                    result.metadata.get("chunk_index", "?"),
                    old_score,
                    result.score,
                    result.score - old_score,
                )

        if not reordered:
            return results[:top_n] if top_n else results

        # Log which results were dropped (present in original but not in rerank output)
        reranked_indices = {
            item["index"]
            for item in rerank_items
            if isinstance(item.get("index"), int)
        }
        for i, result in enumerate(results):
            if i not in reranked_indices:
                logger.info(
                    "Rerank dropped: file=%s chunk=%s  score=%.4f",
                    result.metadata.get("file_name", "?"),
                    result.metadata.get("chunk_index", "?"),
                    result.score,
                )

        if top_n is not None:
            return reordered[:top_n]
        return reordered


class SiliconFlowReranker(APIReranker):
    """SiliconFlow (硅基流动) rerank API — BAAI/bge-reranker-v2-m3 and more."""


class JinaReranker(APIReranker):
    """Jina AI rerank API — jina-reranker-v2-base-multilingual."""


class BgeReranker(Reranker):
    """Reserved extension point for local bge-reranker (sentence-transformers)."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError(
            "bge-reranker 预留：可在此接入 sentence-transformers 或本地推理服务"
        )
