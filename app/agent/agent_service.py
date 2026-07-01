from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

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

if TYPE_CHECKING:
    from app.core.memory.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)

# 注入对话历史时，最多携带的最近轮数
_CONTEXT_MAX_TURNS = 5


class AgentService:
    """Agentic RAG service driven by a LangGraph graph.

    Accepts optional retry dependencies so the retry pass can use forced
    multi-query rewrite and hybrid retrieval without changing the main
    RAGChain configuration.

    When a ``memory`` instance is provided, the service automatically
    persists Q&A pairs to Redis and injects recent conversation context
    into follow-up questions — enabling multi-turn RAG conversations.
    """

    def __init__(
        self,
        rag_chain: RAGChain,
        *,
        retry_retriever: Retriever | HybridRetriever | None = None,
        retry_query_rewriter: QueryRewriter | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.rag_chain = rag_chain
        self.retry_retriever = retry_retriever
        self.retry_query_rewriter = retry_query_rewriter
        self.memory = memory
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
        session_id: str | None = None,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AgentResponse:
        # ── 加载历史 & 注入上下文 ──────────────────────────────
        enriched_question = question
        if session_id and self.memory:
            past_messages = self.memory.load_messages(session_id)
            if past_messages:
                ctx_block = self._build_context_block(past_messages)
                enriched_question = f"{ctx_block}\n\n当前问题: {question}"
                logger.info(
                    "MEM CTX   session=%s  injected %d past messages as context",
                    session_id,
                    len(past_messages),
                )

        # ── 执行 graph ─────────────────────────────────────────
        state = self.graph.invoke(
            {
                "question": enriched_question,
                "top_k": top_k,
                "metadata_filter": metadata_filter,
                "max_iterations": 2,
                "max_top_k": 20,
                "trace_steps": [],
            }
        )

        answer = state.get("answer", NOT_FOUND_ANSWER)

        # ── 持久化本轮对话 ─────────────────────────────────────
        if session_id and self.memory:
            past = self.memory.load_messages(session_id)
            past.append(HumanMessage(content=question))
            past.append(AIMessage(content=answer))
            self.memory.save_messages(session_id, past)

        return AgentResponse(
            answer=answer,
            sources=self._build_sources(state.get("results", [])),
            trace=AgentTrace(
                plan=state.get("plan", PLAN_RAG_SEARCH),
                reflection=state.get("reflection", REFLECTION_INSUFFICIENT_CONTEXT),
                iterations=state.get("iterations", 0),
                steps=[
                    AgentTraceStep(**step) for step in state.get("trace_steps", [])
                ],
            ),
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_block(messages: list, max_turns: int = _CONTEXT_MAX_TURNS) -> str:
        """将最近 N 轮对话历史格式化为上下文文本块，注入到当前问题中。"""
        # 仅保留最近 N 轮（每轮 = human + ai 两条）
        recent = messages[-(max_turns * 2):]
        lines = ["以下是之前的对话历史，请结合上下文理解当前问题：", ""]
        for msg in recent:
            role = "用户" if _msg_type(msg) == "human" else "助手"
            content = _get_msg_content(msg)
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

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


# ------------------------------------------------------------------
# Module helpers
# ------------------------------------------------------------------

def _msg_type(msg) -> str:
    """安全获取消息类型（兼容 dict 和 BaseMessage）。"""
    if hasattr(msg, "type"):
        return msg.type
    return msg.get("type", "") if isinstance(msg, dict) else ""


def _get_msg_content(msg) -> str:
    """安全获取消息内容。"""
    if hasattr(msg, "content"):
        return msg.content or ""
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        return str(content) if content else ""
    return ""
