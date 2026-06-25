from unittest.mock import MagicMock

from app.core.config import Settings
from app.rag.query_rewriter import (
    MultiQueryRewriter,
    RewriteResult,
    SimpleQueryRewriter,
    _extract_section,
    _parse_query_lines,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_extract_section_basic() -> None:
    raw = "---QUERY---\nrewritten text here\n---KEYWORDS---\nkw1 kw2"
    assert _extract_section(raw, "QUERY") == "rewritten text here"
    assert _extract_section(raw, "KEYWORDS") == "kw1 kw2"


def test_extract_section_missing_returns_none() -> None:
    assert _extract_section("no tags here", "QUERY") is None


def test_parse_query_lines_strips_numbering() -> None:
    block = "1. first query\n2. second query\n3. third query"
    result = _parse_query_lines(block)
    assert result == ["first query", "second query", "third query"]


# ---------------------------------------------------------------------------
# SimpleQueryRewriter tests
# ---------------------------------------------------------------------------

def test_simple_rewrite_returns_structured_result() -> None:
    generate = MagicMock(return_value=(
        "---QUERY---\nhow to configure push notifications\n---KEYWORDS---\npush notification configure"
    ))
    rewriter = SimpleQueryRewriter(generate)

    result = rewriter.rewrite("what is push")

    assert isinstance(result, RewriteResult)
    assert result.queries == ["how to configure push notifications"]
    assert result.keywords == ["push", "notification", "configure"]
    generate.assert_called_once()


def test_simple_rewrite_falls_back_on_llm_failure() -> None:
    generate = MagicMock(side_effect=RuntimeError("boom"))
    rewriter = SimpleQueryRewriter(generate)

    result = rewriter.rewrite("important")

    assert result.queries == ["important"]
    assert result.keywords == ["important"]


def test_simple_rewrite_falls_back_on_empty_response() -> None:
    generate = MagicMock(return_value="")
    rewriter = SimpleQueryRewriter(generate)

    result = rewriter.rewrite("test")
    assert result.queries == ["test"]


# ---------------------------------------------------------------------------
# MultiQueryRewriter tests
# ---------------------------------------------------------------------------

_GOOD_MULTI_OUTPUT = (
    "---QUERIES---\n"
    "how to configure push\n"
    "troubleshooting push notifications\n"
    "message delivery failure\n"
    "---KEYWORDS---\n"
    "push notification configure delivery"
)


def test_multi_rewrite_returns_structured_result() -> None:
    generate = MagicMock(return_value=_GOOD_MULTI_OUTPUT)
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("push not working")

    assert isinstance(result, RewriteResult)
    assert len(result.queries) == 3
    assert "how to configure push" in result.queries
    assert result.keywords == ["push", "notification", "configure", "delivery"]


def test_multi_rewrite_strips_numbering_in_queries() -> None:
    raw = (
        "---QUERIES---\n"
        "1. first query\n"
        "2. second query\n"
        "---KEYWORDS---\n"
        "first second"
    )
    generate = MagicMock(return_value=raw)
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("test")
    assert result.queries == ["first query", "second query"]


def test_multi_rewrite_falls_back_on_empty_output() -> None:
    generate = MagicMock(return_value="")
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("test")
    assert result.queries == ["test"]
    assert result.keywords == ["test"]


def test_multi_rewrite_falls_back_on_exception() -> None:
    generate = MagicMock(side_effect=RuntimeError("boom"))
    rewriter = MultiQueryRewriter(generate, num_queries=3)

    result = rewriter.rewrite("test")
    assert result.queries == ["test"]


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
