from __future__ import annotations

import logging
from typing import Any

from app.agent.constants import (
    PLAN_RAG_SEARCH,
    REFLECTION_INSUFFICIENT_CONTEXT,
    REFLECTION_SUPPORTED,
)
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_rewriter import QueryRewriter
from app.rag.rag_chain import NOT_FOUND_ANSWER, RAGChain
from app.rag.retriever import Retriever
from app.rag.vector_store import SearchResult
from app.schemas.agent_schema import AgentResponse, AgentTrace
from app.schemas.chat_schema import Source

logger = logging.getLogger(__name__)


class AgentService:
    """Agentic RAG service driven by a LangGraph graph.

    Accepts optional retry dependencies so the retry pass can use forced
    multi-query rewrite and hybrid retrieval without changing the main
    RAGChain configuration.
    """

    def __init__(
        self,
        rag_chain: RAGChain,
        *,
        retry_retriever: Retriever | HybridRetriever | None = None,
        retry_query_rewriter: QueryRewriter | None = None,
    ) -> None:
        self.rag_chain = rag_chain
        self.retry_retriever = retry_retriever
        self.retry_query_rewriter = retry_query_rewriter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        *,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AgentResponse:
        # 1. Plan
        plan = self._plan(question)

        # 2. Execute
        answer, results = self._execute(
            question, top_k=top_k, metadata_filter=metadata_filter,
        )

        # 3. Reflect
        reflection = self._reflect(answer, results)

        # 4. Final
        return self._final(answer, results, plan, reflection)

    # ------------------------------------------------------------------
    # Plan / Execute / Reflect / Final
    # ------------------------------------------------------------------

    def _plan(self, question: str) -> str:
        """Determine the action plan for the given question.

        First version: always returns 'rag_search'. Future versions can
        inspect the question and select among strategies like direct_answer,
        rewrite_and_search, or need_more_context.
        """
        logger.info("plan=%s", PLAN_RAG_SEARCH)
        return PLAN_RAG_SEARCH

    def _execute(
        self,
        question: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> tuple[str, list[SearchResult]]:
        """Execute the RAG pipeline and return the answer with raw results."""
        answer, results = self.rag_chain.ask(
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        logger.info("source_count=%d", len(results))
        return answer, results

    def _reflect(self, answer: str, results: list[SearchResult]) -> str:
        """Reflect on whether the answer is supported by retrieved context.

        Rules (deterministic, no extra LLM call):
        - 'supported' when sources are non-empty AND the answer is not the
          sentinel 'not found' message.
        - 'insufficient_context' otherwise.
        """
        if results and answer != NOT_FOUND_ANSWER:
            reflection = REFLECTION_SUPPORTED
        else:
            reflection = REFLECTION_INSUFFICIENT_CONTEXT

        logger.info(
            "reflection=%s  source_count=%d  is_not_found=%s",
            reflection,
            len(results),
            answer == NOT_FOUND_ANSWER,
        )
        return reflection

    def _final(
        self,
        answer: str,
        results: list[SearchResult],
        plan: str,
        reflection: str,
    ) -> AgentResponse:
        """Assemble the final AgentResponse with answer, sources, and trace."""
        return AgentResponse(
            answer=answer,
            sources=self._build_sources(results),
            trace=AgentTrace(
                plan=plan,
                reflection=reflection,
                iterations=1,
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sources(results: list[SearchResult]) -> list[Source]:
        """Convert SearchResult list into Source list for the response."""
        sources: list[Source] = []
        for result in results:
            metadata = result.metadata
            sources.append(
                Source(
                    file_name=metadata.get("file_name"),
                    file_path=metadata.get("file_path"),
                    chunk_index=metadata.get("chunk_index"),
                    source_type=metadata.get("source_type"),
                    score=result.score,
                )
            )
        return sources
