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


class TestAgentServiceWithMemory:
    """AgentService + ConversationMemory 集成测试。"""

    def test_ask_without_session_id_skips_memory(self) -> None:
        """不传 session_id 时，不读写 Redis，行为与之前完全一致。"""
        mock_chain = MagicMock()
        mock_chain.ask.return_value = ("回答", [_make_result()])

        mock_memory = MagicMock()
        svc = AgentService(rag_chain=mock_chain, memory=mock_memory)
        resp = svc.ask(question="测试")

        assert resp.session_id is None
        mock_memory.load_messages.assert_not_called()
        mock_memory.save_messages.assert_not_called()

    def test_ask_with_session_id_loads_and_saves(self) -> None:
        """传 session_id 时，加载历史 → 执行 → 持久化。"""
        from langchain_core.messages import AIMessage, HumanMessage

        mock_chain = MagicMock()
        mock_chain.ask.return_value = ("新回答", [_make_result()])

        mock_memory = MagicMock()
        mock_memory.load_messages.return_value = [
            HumanMessage(content="之前的问题"),
            AIMessage(content="之前的回答"),
        ]

        svc = AgentService(rag_chain=mock_chain, memory=mock_memory)
        resp = svc.ask(question="当前问题", session_id="demo-001")

        # 验证加载了历史（被调用两次：构建上下文 + 保存前加载）
        assert mock_memory.load_messages.call_count == 2
        mock_memory.load_messages.assert_any_call("demo-001")
        # 验证问题被注入了上下文
        call_args = mock_chain.ask.call_args
        enriched = call_args.kwargs.get("question") or call_args.args[0]
        assert "之前的问题" in enriched
        assert "当前问题" in enriched
        # 验证持久化：新增了一轮 human + ai
        save_call = mock_memory.save_messages.call_args
        saved_msgs = save_call[0][1]  # args: (session_id, messages)
        assert len(saved_msgs) == 4  # 2 历史 + 2 新
        assert saved_msgs[-2].content == "当前问题"
        assert saved_msgs[-1].content == "新回答"
        # 验证响应
        assert resp.session_id == "demo-001"

    def test_ask_new_session_no_context_injection(self) -> None:
        """全新会话（无历史消息），问题不应被注入上下文前缀。"""
        mock_chain = MagicMock()
        mock_chain.ask.return_value = ("回答", [_make_result()])

        mock_memory = MagicMock()
        mock_memory.load_messages.return_value = []  # 无历史

        svc = AgentService(rag_chain=mock_chain, memory=mock_memory)
        svc.ask(question="新会话问题", session_id="new-session")

        call_args = mock_chain.ask.call_args
        enriched = call_args.kwargs.get("question") or call_args.args[0]
        # 新会话不应有 "以下是之前的对话历史"
        assert "以下是之前的对话历史" not in enriched
        assert enriched == "新会话问题"

    def test_agent_service_accepts_memory(self) -> None:
        """构造器应接受并保存 memory 实例。"""
        mock_memory = MagicMock()
        svc = AgentService(rag_chain=MagicMock(), memory=mock_memory)
        assert svc.memory is mock_memory

    def test_context_block_truncates_to_max_turns(self) -> None:
        """_build_context_block 仅保留最近 N 轮。"""
        from langchain_core.messages import AIMessage, HumanMessage

        # 10 轮 = 20 条消息，max_turns=5 → 应只保留最后 10 条
        msgs = []
        for i in range(10):
            msgs.append(HumanMessage(content=f"Q{i}"))
            msgs.append(AIMessage(content=f"A{i}"))

        block = AgentService._build_context_block(msgs, max_turns=5)
        # 最早的消息不应出现
        assert "Q0" not in block
        assert "A0" not in block
        # 最近的消息应出现
        assert "Q9" in block
        assert "A9" in block


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
