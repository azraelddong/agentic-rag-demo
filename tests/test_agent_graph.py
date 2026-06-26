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
