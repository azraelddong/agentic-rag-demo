"""Unit tests for AgentService driven by LangGraph."""

from unittest.mock import MagicMock

from app.agent.agent_service import AgentService
from app.agent.constants import (
    REFLECTION_INSUFFICIENT_CONTEXT,
    REFLECTION_SUPPORTED,
)
from app.rag.rag_chain import NOT_FOUND_ANSWER
from app.rag.vector_store import SearchResult


def _make_result(text: str = "some chunk", score: float = 0.9) -> SearchResult:
    return SearchResult(
        text=text,
        score=score,
        metadata={
            "file_name": "test.md",
            "file_path": "/docs/test.md",
            "chunk_index": 0,
            "source_type": "md",
        },
    )


class TestAgentTraceSchema:
    def test_trace_can_include_steps(self) -> None:
        from app.schemas.agent_schema import AgentTrace, AgentTraceStep

        trace = AgentTrace(
            plan="rag_search",
            reflection="supported",
            iterations=1,
            steps=[
                AgentTraceStep(node="plan", decision="rag_search"),
                AgentTraceStep(node="execute_rag", source_count=1, top_k=5),
            ],
        )

        assert trace.steps[0].node == "plan"
        assert trace.steps[0].decision == "rag_search"
        assert trace.steps[1].source_count == 1
        assert trace.steps[1].top_k == 5


class TestAgentServiceAsk:
    """End-to-end ask() with a mocked RAGChain, driven by LangGraph."""

    def test_ask_supported_flow(self) -> None:
        mock_chain = MagicMock()
        mock_chain.ask.return_value = ("这是一个回答", [_make_result()])

        svc = AgentService(rag_chain=mock_chain)
        resp = svc.ask(question="测试问题")

        assert mock_chain.ask.call_count == 1
        assert resp.answer == "这是一个回答"
        assert len(resp.sources) == 1
        assert resp.trace.plan == "rag_search"
        assert resp.trace.reflection == REFLECTION_SUPPORTED
        assert resp.trace.iterations == 1
        assert len(resp.trace.steps) > 0

    def test_ask_insufficient_context_flow(self) -> None:
        mock_chain = MagicMock()
        mock_chain.settings.rag_top_k = 5
        mock_chain.ask.return_value = (NOT_FOUND_ANSWER, [])

        svc = AgentService(rag_chain=mock_chain)
        resp = svc.ask(question="无匹配问题")

        assert resp.answer == NOT_FOUND_ANSWER
        assert len(resp.sources) == 0
        assert resp.trace.reflection == REFLECTION_INSUFFICIENT_CONTEXT
        assert len(resp.trace.steps) > 0

    def test_agent_service_accepts_retry_dependencies(self) -> None:
        rag_chain = MagicMock()
        retry_retriever = MagicMock()
        retry_query_rewriter = MagicMock()

        svc = AgentService(
            rag_chain=rag_chain,
            retry_retriever=retry_retriever,
            retry_query_rewriter=retry_query_rewriter,
        )

        assert svc.rag_chain is rag_chain
        assert svc.retry_retriever is retry_retriever
        assert svc.retry_query_rewriter is retry_query_rewriter

    def test_ask_passes_top_k_and_filter(self) -> None:
        mock_chain = MagicMock()
        mock_chain.ask.return_value = ("ok", [_make_result()])

        svc = AgentService(rag_chain=mock_chain)
        svc.ask(question="x", top_k=3, metadata_filter={"source_type": "pdf"})

        call_kwargs = mock_chain.ask.call_args.kwargs
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["metadata_filter"] == {"source_type": "pdf"}

    def test_ask_retries_once_when_context_is_insufficient(self) -> None:
        mock_chain = MagicMock()
        mock_chain.ask.side_effect = [
            (NOT_FOUND_ANSWER, []),
            ("第二轮回答", [_make_result()]),
        ]

        retry_retriever = MagicMock()
        retry_query_rewriter = MagicMock()

        svc = AgentService(
            rag_chain=mock_chain,
            retry_retriever=retry_retriever,
            retry_query_rewriter=retry_query_rewriter,
        )

        resp = svc.ask(question="测试问题", top_k=5)

        assert resp.answer == "第二轮回答"
        assert resp.trace.iterations == 2
        assert resp.trace.reflection == REFLECTION_SUPPORTED
        assert [step.node for step in resp.trace.steps] == [
            "plan",
            "execute_rag",
            "reflect",
            "prepare_retry",
            "execute_rag",
            "reflect",
            "final",
        ]

        retry_call = mock_chain.ask.call_args_list[1]
        assert retry_call.kwargs["top_k"] == 10
        assert retry_call.kwargs["retriever_override"] is retry_retriever
        assert retry_call.kwargs["query_rewriter_override"] is retry_query_rewriter
        assert retry_call.kwargs["force_query_rewrite"] is True
