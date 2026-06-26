from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

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


class AgentState(TypedDict, total=False):
    question: str
    top_k: int | None
    metadata_filter: dict[str, Any] | None
    max_iterations: int
    max_top_k: int
    plan: str
    answer: str
    results: list[SearchResult]
    reflection: str
    iterations: int
    current_top_k: int | None
    retry_reason: str | None
    use_retry_strategy: bool
    trace_steps: list[dict[str, Any]]


def _append_step(state: AgentState, step: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("trace_steps", []), step]


def build_agentic_rag_graph(
    *,
    rag_chain: RAGChain,
    retry_retriever: Retriever | HybridRetriever | None = None,
    retry_query_rewriter: QueryRewriter | None = None,
):
    def plan_node(state: AgentState) -> AgentState:
        return {
            "plan": PLAN_RAG_SEARCH,
            "iterations": state.get("iterations", 0),
            "current_top_k": state.get("top_k"),
            "use_retry_strategy": False,
            "trace_steps": _append_step(
                state,
                {"node": "plan", "decision": PLAN_RAG_SEARCH},
            ),
        }

    def execute_rag_node(state: AgentState) -> AgentState:
        use_retry = state.get("use_retry_strategy", False)
        ask_kwargs: dict[str, Any] = {
            "top_k": state.get("current_top_k"),
            "metadata_filter": state.get("metadata_filter"),
        }
        if use_retry:
            ask_kwargs.update(
                {
                    "retriever_override": retry_retriever,
                    "query_rewriter_override": retry_query_rewriter,
                    "force_query_rewrite": True,
                }
            )

        answer, results = rag_chain.ask(state["question"], **ask_kwargs)
        iterations = state.get("iterations", 0) + 1

        return {
            "answer": answer,
            "results": results,
            "iterations": iterations,
            "trace_steps": _append_step(
                state,
                {
                    "node": "execute_rag",
                    "source_count": len(results),
                    "top_k": state.get("current_top_k"),
                    "decision": "retry" if use_retry else "initial",
                },
            ),
        }

    def reflect_node(state: AgentState) -> AgentState:
        results = state.get("results", [])
        answer = state.get("answer", "")
        if results and answer != NOT_FOUND_ANSWER:
            reflection = REFLECTION_SUPPORTED
        else:
            reflection = REFLECTION_INSUFFICIENT_CONTEXT

        return {
            "reflection": reflection,
            "trace_steps": _append_step(
                state,
                {"node": "reflect", "reflection": reflection},
            ),
        }

    def route_after_reflect(state: AgentState) -> Literal["prepare_retry", "final"]:
        if (
            state.get("reflection") == REFLECTION_INSUFFICIENT_CONTEXT
            and state.get("iterations", 0) < state.get("max_iterations", 2)
        ):
            return "prepare_retry"
        return "final"

    def prepare_retry_node(state: AgentState) -> AgentState:
        base_top_k = state.get("top_k") or rag_chain.settings.rag_top_k
        max_top_k = state.get("max_top_k", 20)
        retry_top_k = min(base_top_k * 2, max_top_k)
        retry_reason = "insufficient_context"

        return {
            "current_top_k": retry_top_k,
            "retry_reason": retry_reason,
            "use_retry_strategy": True,
            "trace_steps": _append_step(
                state,
                {
                    "node": "prepare_retry",
                    "retry_reason": retry_reason,
                    "top_k": retry_top_k,
                    "decision": "expanded_rewrite_hybrid_retry",
                },
            ),
        }

    def final_node(state: AgentState) -> AgentState:
        return {
            "trace_steps": _append_step(
                state,
                {
                    "node": "final",
                    "reflection": state.get("reflection"),
                    "source_count": len(state.get("results", [])),
                },
            )
        }

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("execute_rag", execute_rag_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("prepare_retry", prepare_retry_node)
    builder.add_node("final", final_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute_rag")
    builder.add_edge("execute_rag", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "prepare_retry": "prepare_retry",
            "final": "final",
        },
    )
    builder.add_edge("prepare_retry", "execute_rag")
    builder.add_edge("final", END)

    return builder.compile()
