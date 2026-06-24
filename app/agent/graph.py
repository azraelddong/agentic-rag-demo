from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Reserved LangGraph state for future Agentic RAG workflows."""

    query: str
    rewritten_query: str
    route: str
    contexts: list[dict[str, Any]]
    answer: str
    judge_result: dict[str, Any]


def build_agentic_rag_graph() -> Any:
    """Build the future LangGraph graph.

    Phase 1 intentionally keeps only Basic RAG. Suggested future nodes:
    - query rewrite
    - retrieval router
    - multi-hop retrieval
    - answer judge
    - self-correction
    """

    raise NotImplementedError(
        "LangGraph 骨架已预留。第一阶段只实现基础 RAG，后续可在此组装 Agentic RAG 图。"
    )
