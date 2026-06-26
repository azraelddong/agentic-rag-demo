from __future__ import annotations

from hashlib import sha256
import logging
from typing import Any

from app.core.config import Settings
from app.llm.chat_model import ChatModel
from app.rag.prompt_builder import PromptBuilder
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_rewriter import QueryRewriter, RewriteResult
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = "知识库中未找到相关信息"

"""RAG链，负责协调检索、重排序、构建提示词和生成等组件，实现基于检索增强的问答功能。"""
class RAGChain:
    """Basic retrieval-augmented generation chain for phase 1."""

    def __init__(
        self,
        *,
        settings: Settings,
        retriever: Retriever | HybridRetriever,
        prompt_builder: PromptBuilder,
        chat_model: ChatModel,
        reranker: Reranker,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.chat_model = chat_model
        self.reranker = reranker
        self.query_rewriter = query_rewriter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        retriever_override: Retriever | HybridRetriever | None = None,
        query_rewriter_override: QueryRewriter | None = None,
        force_query_rewrite: bool | None = None,
    ) -> tuple[str, list[SearchResult]]:
        normalized_question = question.strip()

        # Select active retriever and rewriter (allow per-call overrides)
        active_retriever = retriever_override or self.retriever
        active_query_rewriter = query_rewriter_override or self.query_rewriter
        rewrite_enabled = (
            force_query_rewrite
            if force_query_rewrite is not None
            else self.settings.query_rewrite_enabled
        )

        # Determine retrieval budget and rerank target
        if self.settings.rerank_enabled:
            retrieval_k = top_k or self.settings.rerank_retrieval_k
            rerank_top_n = self.settings.rerank_top_n
        else:
            retrieval_k = top_k or self.settings.rag_top_k
            rerank_top_n = None

        # --- Query rewrite step ------------------------------------------------
        rewrite_result = RewriteResult(queries=[normalized_question], keywords=[])
        if rewrite_enabled and active_query_rewriter is not None:
            logger.info("Query rewrite 前: \"%s\"", normalized_question)
            rewrite_result = active_query_rewriter.rewrite(normalized_question)
            logger.info(
                "Query rewrite 后: queries=%s  keywords=%s",
                rewrite_result.queries, rewrite_result.keywords,
            )

        # --- Multi-retrieval + dedup step --------------------------------------
        # Score filtering is done inside the retriever (pre-RRF for hybrid,
        # COSINE threshold for dense) — no separate _filter_by_score needed.
        results = self._multi_retrieve(
            active_retriever,
            rewrite_result.queries,
            retrieval_k,
            metadata_filter,
            keywords=rewrite_result.keywords,
            score_threshold=self.settings.rag_score_threshold,
        )
        self._log_retrieval_results(results)
        results = self.reranker.rerank(normalized_question, results, top_n=rerank_top_n)
        self._log_reranked_results(results)

        if not results:
            logger.info("No relevant chunks found for query")
            return NOT_FOUND_ANSWER, []

        messages = self.prompt_builder.build_messages(normalized_question, results)
        answer = self.chat_model.generate(messages)
        if not answer:
            return NOT_FOUND_ANSWER, results
        return answer, results

    # ------------------------------------------------------------------
    # Multi-retrieval helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_fingerprint(result: SearchResult) -> str:
        """Stable dedup key for a chunk (first 200 chars hash)."""
        return sha256(result.text[:200].encode()).hexdigest()

    def _multi_retrieve(
        self,
        retriever: Retriever | HybridRetriever,
        queries: list[str],
        retrieval_k: int,
        metadata_filter: dict[str, Any] | None,
        *,
        keywords: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Run retrieval for each query, then deduplicate keeping the best score."""
        if len(queries) == 1:
            return retriever.retrieve(
                queries[0], top_k=retrieval_k, metadata_filter=metadata_filter,
                keywords=keywords, score_threshold=score_threshold,
            )

        # Distribute retrieval budget across queries
        per_query_k = max(3, retrieval_k // len(queries))
        logger.info(
            "Multi-Query 检索：%d 个查询，每个检索 %d 条",
            len(queries), per_query_k,
        )

        seen: dict[str, SearchResult] = {}
        for query in queries:
            batch = retriever.retrieve(
                query, top_k=per_query_k, metadata_filter=metadata_filter,
                keywords=keywords, score_threshold=score_threshold,
            )
            for result in batch:
                fp = self._chunk_fingerprint(result)
                if fp not in seen or result.score > seen[fp].score:
                    seen[fp] = result

        merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        logger.info(
            "Multi-Query 去重后：%d 条 → %d 条（合并前 %d 条）",
            len(queries) * per_query_k, len(merged), sum(1 for _ in seen),
        )
        return merged[:retrieval_k]

    def _filter_by_score(self, results: list[SearchResult]) -> list[SearchResult]:
        threshold = self.settings.rag_score_threshold
        if threshold is None:
            return results

        metric_type = self.settings.milvus_metric_type.upper()
        if metric_type == "L2":
            return [result for result in results if result.score <= threshold]
        return [result for result in results if result.score >= threshold]

    # ------------------------------------------------------------------
    # Detailed logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_result(index: int, result: SearchResult) -> str:
        """Format a single SearchResult as a compact one-line summary."""
        meta = result.metadata
        text_preview = result.text[:120].replace("\n", " ")
        return (
            f"[{index:>2}] score={result.score:.4f}  "
            f"file={meta.get('file_name', '?')}  "
            f"chunk={meta.get('chunk_index', '?')}  "
            f"text=\"{text_preview}...\""
        )

    def _log_retrieval_results(self, results: list[SearchResult]) -> None:
        """Log retrieved results after score filtering, before reranking."""
        if not results:
            logger.info("检索结果为空（score 过滤后无剩余）")
            return

        header = f"检索结果（共 {len(results)} 条，向量相似度排序）"
        logger.info(header)
        for i, result in enumerate(results):
            logger.info(self._format_result(i, result))

    def _log_reranked_results(self, results: list[SearchResult]) -> None:
        """Log results after reranking."""
        if not results:
            logger.info("Rerank 结果为空")
            return

        rerank_active = self.settings.rerank_enabled
        label = "Rerank 后结果" if rerank_active else "最终结果（未启用 Rerank）"
        header = f"{label}（共 {len(results)} 条）"
        logger.info(header)
        for i, result in enumerate(results):
            logger.info(self._format_result(i, result))
