from __future__ import annotations

import logging
from typing import Any

from app.agent.constants import (
    PLAN_RAG_SEARCH,
    REFLECTION_INSUFFICIENT_CONTEXT,
    REFLECTION_SUPPORTED,
)
from app.agent.graph import build_agentic_rag_graph
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_rewriter import QueryRewriter
from app.rag.rag_chain import NOT_FOUND_ANSWER, RAGChain
from app.rag.retriever import Retriever
from app.rag.vector_store import SearchResult
from app.schemas.agent_schema import AgentResponse, AgentTrace, AgentTraceStep
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
        self.graph = build_agentic_rag_graph(
            rag_chain=rag_chain,
            retry_retriever=retry_retriever,
            retry_query_rewriter=retry_query_rewriter,
        )

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
        state = self.graph.invoke(
            {
                "question": question,
                "top_k": top_k,
                "metadata_filter": metadata_filter,
                "max_iterations": 2,
                "max_top_k": 20,
                "trace_steps": [],
            }
        )

        return AgentResponse(
            answer=state.get("answer", NOT_FOUND_ANSWER),
            sources=self._build_sources(state.get("results", [])),
            trace=AgentTrace(
                plan=state.get("plan", PLAN_RAG_SEARCH),
                reflection=state.get("reflection", REFLECTION_INSUFFICIENT_CONTEXT),
                iterations=state.get("iterations", 0),
                steps=[
                    AgentTraceStep(**step) for step in state.get("trace_steps", [])
                ],
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
