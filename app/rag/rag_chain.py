from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings
from app.llm.chat_model import ChatModel
from app.rag.prompt_builder import PromptBuilder
from app.rag.query_rewriter import QueryRewriter
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
        retriever: Retriever,
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

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> tuple[str, list[SearchResult]]:
        normalized_question = question.strip()

        # Rewrite query for retrieval if enabled (keep original for LLM prompt)
        search_query = normalized_question
        if self.settings.query_rewrite_enabled and self.query_rewriter is not None:
            logger.info("Query rewrite 前: \"%s\"", normalized_question)
            search_query = self.query_rewriter.rewrite(normalized_question)
            logger.info("Query rewrite 后: \"%s\"", search_query)

        # Retrieve more candidates when reranking is enabled (M in M→N strategy)
        if self.settings.rerank_enabled:
            retrieval_k = top_k or self.settings.rerank_retrieval_k
            rerank_top_n = self.settings.rerank_top_n
        else:
            retrieval_k = top_k or self.settings.rag_top_k
            rerank_top_n = None

        results = self.retriever.retrieve(
            search_query,
            top_k=retrieval_k,
            metadata_filter=metadata_filter,
        )
        results = self._filter_by_score(results)
        self._log_retrieval_results(results)
        results = self.reranker.rerank(search_query, results, top_n=rerank_top_n)
        self._log_reranked_results(results)

        if not results:
            logger.info("No relevant chunks found for query")
            return NOT_FOUND_ANSWER, []

        # Use the original question for the LLM, not the rewritten search query
        messages = self.prompt_builder.build_messages(normalized_question, results)
        answer = self.chat_model.generate(messages)
        if not answer:
            return NOT_FOUND_ANSWER, results
        return answer, results

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
