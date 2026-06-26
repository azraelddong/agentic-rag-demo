"""Unit tests for AgentService – plan / execute / reflect / final."""

from unittest.mock import MagicMock

import pytest

from app.agent.agent_service import (
    PLAN_RAG_SEARCH,
    REFLECTION_INSUFFICIENT_CONTEXT,
    REFLECTION_SUPPORTED,
    AgentService,
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


class TestAgentServicePlan:
    """Plan always returns 'rag_search' in the first version."""

    def test_plan_returns_rag_search(self) -> None:
        svc = AgentService(rag_chain=MagicMock())
        assert svc._plan("任意问题") == PLAN_RAG_SEARCH


class TestAgentServiceReflect:
    """Reflect uses deterministic rules — no extra LLM call."""

    def test_supported_when_sources_and_valid_answer(self) -> None:
        svc = AgentService(rag_chain=MagicMock())
        results = [_make_result()]
        assert svc._reflect("有意义的回答", results) == REFLECTION_SUPPORTED

    def test_insufficient_context_when_no_sources(self) -> None:
        svc = AgentService(rag_chain=MagicMock())
        assert svc._reflect("有意义的回答", []) == REFLECTION_INSUFFICIENT_CONTEXT

    def test_insufficient_context_when_not_found_answer(self) -> None:
        svc = AgentService(rag_chain=MagicMock())
        results = [_make_result()]
        assert svc._reflect(NOT_FOUND_ANSWER, results) == REFLECTION_INSUFFICIENT_CONTEXT

    def test_insufficient_context_when_no_sources_and_not_found(self) -> None:
        svc = AgentService(rag_chain=MagicMock())
        assert svc._reflect(NOT_FOUND_ANSWER, []) == REFLECTION_INSUFFICIENT_CONTEXT


class TestAgentServiceFinal:
    """Final assembles answer + sources + trace correctly."""

    def test_response_contains_answer_sources_trace(self) -> None:
        svc = AgentService(rag_chain=MagicMock())
        results = [_make_result(score=0.85)]
        response = svc._final("答案", results, PLAN_RAG_SEARCH, REFLECTION_SUPPORTED)

        assert response.answer == "答案"
        assert len(response.sources) == 1
        assert response.sources[0].file_name == "test.md"
        assert response.sources[0].score == 0.85
        assert response.trace.plan == PLAN_RAG_SEARCH
        assert response.trace.reflection == REFLECTION_SUPPORTED
        assert response.trace.iterations == 1


def test_trace_can_include_steps() -> None:
    from app.schemas.agent_schema import AgentTrace, AgentTraceStep

    trace = AgentTrace(
        plan="rag_search",
        reflection="supported",
        iterations=1,
        steps=[
            AgentTraceStep(
                node="plan",
                decision="rag_search",
            ),
            AgentTraceStep(
                node="execute_rag",
                source_count=1,
                top_k=5,
            ),
        ],
    )

    assert trace.steps[0].node == "plan"
    assert trace.steps[0].decision == "rag_search"
    assert trace.steps[1].source_count == 1
    assert trace.steps[1].top_k == 5


class TestAgentServiceAsk:
    """End-to-end ask() with a mocked RAGChain."""

    def test_ask_supported_flow(self) -> None:
        mock_chain = MagicMock()
        mock_chain.ask.return_value = ("这是一个回答", [_make_result()])

        svc = AgentService(rag_chain=mock_chain)
        resp = svc.ask(question="测试问题")

        mock_chain.ask.assert_called_once()
        assert resp.answer == "这是一个回答"
        assert len(resp.sources) == 1
        assert resp.trace.plan == PLAN_RAG_SEARCH
        assert resp.trace.reflection == REFLECTION_SUPPORTED
        assert resp.trace.iterations == 1

    def test_ask_insufficient_context_flow(self) -> None:
        mock_chain = MagicMock()
        mock_chain.ask.return_value = (NOT_FOUND_ANSWER, [])

        svc = AgentService(rag_chain=mock_chain)
        resp = svc.ask(question="无匹配问题")

        assert resp.answer == NOT_FOUND_ANSWER
        assert len(resp.sources) == 0
        assert resp.trace.reflection == REFLECTION_INSUFFICIENT_CONTEXT

    def test_ask_passes_top_k_and_filter(self) -> None:
        mock_chain = MagicMock()
        mock_chain.ask.return_value = ("ok", [_make_result()])

        svc = AgentService(rag_chain=mock_chain)
        svc.ask(question="x", top_k=3, metadata_filter={"source_type": "pdf"})

        call_kwargs = mock_chain.ask.call_args.kwargs
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["metadata_filter"] == {"source_type": "pdf"}
