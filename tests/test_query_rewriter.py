from unittest.mock import MagicMock

from app.core.config import Settings
from app.rag.query_rewriter import (
    MultiQueryRewriter,
    SimpleQueryRewriter,
)


# ---------------------------------------------------------------------------
# SimpleQueryRewriter tests
# ---------------------------------------------------------------------------

def test_simple_rewrite_returns_single_query() -> None:
    generate = MagicMock(return_value="how to configure push notifications")
    rewriter = SimpleQueryRewriter(generate)

    result = rewriter.rewrite("what is push")

    assert result == ["how to configure push notifications"]
    generate.assert_called_once()


def test_simple_rewrite_falls_back_when_llm_returns_same_text() -> None:
    generate = MagicMock(return_value="same question")
    rewriter = SimpleQueryRewriter(generate)

    result = rewriter.rewrite("same question")
    assert result == ["same question"]


def test_simple_rewrite_falls_back_when_output_is_empty() -> None:
    generate = MagicMock(return_value="")
    rewriter = SimpleQueryRewriter(generate)

    result = rewriter.rewrite("test")
    assert result == ["test"]


def test_simple_rewrite_falls_back_on_exception() -> None:
    generate = MagicMock(side_effect=RuntimeError("boom"))
    rewriter = SimpleQueryRewriter(generate)

    result = rewriter.rewrite("important")
    assert result == ["important"]


def test_simple_rewrite_maps_langchain_human_role_to_user() -> None:
    generate = MagicMock(return_value="rewritten")
    rewriter = SimpleQueryRewriter(generate)

    rewriter.rewrite("test")

    messages = generate.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# MultiQueryRewriter tests
# ---------------------------------------------------------------------------

def test_multi_rewrite_returns_multiple_queries() -> None:
    generate = MagicMock(return_value="how to configure push\ntroubleshooting push notifications\nmessage delivery failure")
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("push not working")

    assert len(result) == 3
    assert "how to configure push" in result
    generate.assert_called_once()


def test_multi_rewrite_strips_numbering_prefixes() -> None:
    generate = MagicMock(return_value="1. first query\n2. second query\n3. third query")
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("test")

    assert result == ["first query", "second query", "third query"]


def test_multi_rewrite_strips_bullet_prefixes() -> None:
    generate = MagicMock(return_value="- first query\n- second query\n- third query")
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("test")

    assert result == ["first query", "second query", "third query"]


def test_multi_rewrite_truncates_to_num_queries() -> None:
    generate = MagicMock(return_value="a\nb\nc\nd\ne\nf")
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("test")
    assert len(result) == 3


def test_multi_rewrite_falls_back_on_empty_response() -> None:
    generate = MagicMock(return_value="")
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("test")
    assert result == ["test"]


def test_multi_rewrite_falls_back_on_exception() -> None:
    generate = MagicMock(side_effect=RuntimeError("boom"))
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("test")
    assert result == ["test"]


def test_multi_rewrite_skips_empty_lines() -> None:
    generate = MagicMock(return_value="first\n\n\nsecond\n")
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("test")
    assert result == ["first", "second"]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_query_rewrite_method_defaults_to_none() -> None:
    settings = Settings(query_rewrite_method="none")
    assert settings.query_rewrite_method == "none"
    assert settings.query_rewrite_enabled is False


def test_query_rewrite_enabled_property() -> None:
    assert Settings(query_rewrite_method="none").query_rewrite_enabled is False
    assert Settings(query_rewrite_method="simple").query_rewrite_enabled is True
    assert Settings(query_rewrite_method="multi").query_rewrite_enabled is True


def test_query_rewrite_multi_count_default() -> None:
    assert Settings(query_rewrite_multi_count=3).query_rewrite_multi_count == 3
    assert Settings(query_rewrite_multi_count=5).query_rewrite_multi_count == 5
