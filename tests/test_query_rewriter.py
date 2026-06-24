from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.rag.query_rewriter import QueryRewriter


# ---------------------------------------------------------------------------
# QueryRewriter tests
# ---------------------------------------------------------------------------

def test_rewrite_returns_llm_output() -> None:
    """The rewriter returns the LLM's output when it differs from input."""
    generate = MagicMock(return_value="如何配置企业微信的消息推送功能")
    rewriter = QueryRewriter(generate)

    result = rewriter.rewrite("企业微信怎么推送消息")

    assert result == "如何配置企业微信的消息推送功能"
    generate.assert_called_once()


def test_rewrite_falls_back_when_output_equals_input() -> None:
    """When the LLM returns the exact same text, keep the original."""
    generate = MagicMock(return_value="same question")
    rewriter = QueryRewriter(generate)

    result = rewriter.rewrite("same question")
    assert result == "same question"


def test_rewrite_falls_back_when_output_is_empty() -> None:
    """When the LLM returns empty string, keep the original."""
    generate = MagicMock(return_value="")
    rewriter = QueryRewriter(generate)

    result = rewriter.rewrite("test question")
    assert result == "test question"


def test_rewrite_falls_back_on_exception() -> None:
    """When the LLM raises, the rewriter returns the original question."""
    generate = MagicMock(side_effect=RuntimeError("LLM down"))
    rewriter = QueryRewriter(generate)

    result = rewriter.rewrite("important question")
    assert result == "important question"


def test_rewrite_prompt_includes_original_question() -> None:
    """The prompt sent to the LLM contains the user's original question."""
    generate = MagicMock(return_value="rewritten")
    rewriter = QueryRewriter(generate)

    rewriter.rewrite("什么是 RAG")

    messages = generate.call_args[0][0]
    user_content = messages[1]["content"]
    assert "什么是 RAG" in user_content


def test_rewrite_maps_langchain_human_role_to_user() -> None:
    """LangChain HumanMessage.type='human' must be mapped to role='user' for OpenAI API."""
    generate = MagicMock(return_value="rewritten")
    rewriter = QueryRewriter(generate)

    rewriter.rewrite("test")

    messages = generate.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"  # not 'human'


def test_query_rewrite_disabled_by_default() -> None:
    """The config defaults to query_rewrite_enabled=False."""
    settings = Settings(query_rewrite_enabled=False)
    assert settings.query_rewrite_enabled is False


def test_query_rewrite_can_be_enabled() -> None:
    settings = Settings(query_rewrite_enabled=True)
    assert settings.query_rewrite_enabled is True
