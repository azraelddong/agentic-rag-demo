from unittest.mock import MagicMock

from app.core.config import Settings
from app.rag.rag_chain import RAGChain
from app.rag.reranker import NoopReranker
from app.rag.vector_store import SearchResult


class DummyPromptBuilder:
    def build_messages(self, question, results):
        return [{"role": "user", "content": question}]


class DummyChatModel:
    def generate(self, messages):
        return "generated answer"


def _result(score: float = 0.9) -> SearchResult:
    return SearchResult(
        text="chunk text",
        score=score,
        metadata={
            "file_name": "doc.md",
            "file_path": "/docs/doc.md",
            "chunk_index": 0,
            "source_type": "md",
        },
    )


def test_rag_chain_uses_retriever_override_when_provided() -> None:
    default_retriever = MagicMock()
    default_retriever.retrieve.return_value = []

    override_retriever = MagicMock()
    override_retriever.retrieve.return_value = [_result()]

    chain = RAGChain(
        settings=Settings(query_rewrite_method="none"),
        retriever=default_retriever,
        prompt_builder=DummyPromptBuilder(),
        chat_model=DummyChatModel(),
        reranker=NoopReranker(),
        query_rewriter=None,
    )

    answer, results = chain.ask(
        "question",
        top_k=3,
        retriever_override=override_retriever,
    )

    assert answer == "generated answer"
    assert len(results) == 1
    default_retriever.retrieve.assert_not_called()
    override_retriever.retrieve.assert_called_once()


def test_rag_chain_can_force_query_rewriter_override() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = [_result()]

    rewriter = MagicMock()
    rewriter.rewrite.return_value.queries = ["rewritten question"]
    rewriter.rewrite.return_value.keywords = ["keyword"]

    chain = RAGChain(
        settings=Settings(query_rewrite_method="none"),
        retriever=retriever,
        prompt_builder=DummyPromptBuilder(),
        chat_model=DummyChatModel(),
        reranker=NoopReranker(),
        query_rewriter=None,
    )

    chain.ask(
        "original question",
        top_k=3,
        query_rewriter_override=rewriter,
        force_query_rewrite=True,
    )

    rewriter.rewrite.assert_called_once_with("original question")
    retriever.retrieve.assert_called_once()
    assert retriever.retrieve.call_args.args[0] == "rewritten question"


# ---------------------------------------------------------------------------
# LangGraph graph behavior tests
# ---------------------------------------------------------------------------

from app.agent.constants import (
    PLAN_RAG_SEARCH,
    REFLECTION_INSUFFICIENT_CONTEXT,
    REFLECTION_SUPPORTED,
)
from app.agent.graph import build_agentic_rag_graph
from app.rag.rag_chain import NOT_FOUND_ANSWER


def test_graph_finishes_after_supported_first_pass() -> None:
    rag_chain = MagicMock()
    rag_chain.ask.return_value = ("answer", [_result()])

    graph = build_agentic_rag_graph(
        rag_chain=rag_chain,
        retry_retriever=MagicMock(),
        retry_query_rewriter=MagicMock(),
    )

    state = graph.invoke(
        {
            "question": "q",
            "top_k": 5,
            "metadata_filter": None,
            "max_iterations": 2,
            "max_top_k": 20,
            "trace_steps": [],
        }
    )

    assert state["answer"] == "answer"
    assert state["reflection"] == REFLECTION_SUPPORTED
    assert state["iterations"] == 1
    assert rag_chain.ask.call_count == 1
    assert [step["node"] for step in state["trace_steps"]] == [
        "plan",
        "execute_rag",
        "reflect",
        "final",
    ]


def test_graph_retries_once_with_enhanced_strategy() -> None:
    rag_chain = MagicMock()
    rag_chain.ask.side_effect = [
        (NOT_FOUND_ANSWER, []),
        ("retry answer", [_result()]),
    ]

    retry_retriever = MagicMock()
    retry_query_rewriter = MagicMock()

    graph = build_agentic_rag_graph(
        rag_chain=rag_chain,
        retry_retriever=retry_retriever,
        retry_query_rewriter=retry_query_rewriter,
    )

    state = graph.invoke(
        {
            "question": "q",
            "top_k": 5,
            "metadata_filter": None,
            "max_iterations": 2,
            "max_top_k": 20,
            "trace_steps": [],
        }
    )

    assert state["answer"] == "retry answer"
    assert state["reflection"] == REFLECTION_SUPPORTED
    assert state["iterations"] == 2
    assert rag_chain.ask.call_count == 2

    retry_call = rag_chain.ask.call_args_list[1]
    assert retry_call.kwargs["top_k"] == 10
    assert retry_call.kwargs["retriever_override"] is retry_retriever
    assert retry_call.kwargs["query_rewriter_override"] is retry_query_rewriter
    assert retry_call.kwargs["force_query_rewrite"] is True
    assert "prepare_retry" in [step["node"] for step in state["trace_steps"]]


def test_graph_stops_after_retry_failure() -> None:
    rag_chain = MagicMock()
    rag_chain.ask.return_value = (NOT_FOUND_ANSWER, [])

    graph = build_agentic_rag_graph(
        rag_chain=rag_chain,
        retry_retriever=MagicMock(),
        retry_query_rewriter=MagicMock(),
    )

    state = graph.invoke(
        {
            "question": "q",
            "top_k": 5,
            "metadata_filter": None,
            "max_iterations": 2,
            "max_top_k": 20,
            "trace_steps": [],
        }
    )

    assert state["reflection"] == REFLECTION_INSUFFICIENT_CONTEXT
    assert state["iterations"] == 2
    assert rag_chain.ask.call_count == 2
