from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from app.llm.chat_model import ChatMessage

logger = logging.getLogger(__name__)

# LangChain message .type → OpenAI-compatible role
_LC_TYPE_TO_ROLE: dict[str, str] = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
}


@dataclass
class RewriteResult:
    """Structured output from query rewriting.

    ``queries`` — rewritten queries for dense / semantic search.
    ``keywords`` — extracted content-bearing terms for BM25 keyword search.
    """

    queries: list[str]
    keywords: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.queries)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class QueryRewriter(ABC):
    """Abstract base for query rewriting strategies."""

    @abstractmethod
    def rewrite(self, question: str) -> RewriteResult:
        """Return rewritten queries + extracted keywords."""


# ---------------------------------------------------------------------------
# Simple (single-query) rewriter
# ---------------------------------------------------------------------------

_SIMPLE_PROMPT = """你是一个查询优化助手。将用户原始问题改写为更适合知识库向量检索的查询语句，并提取用于关键词检索的核心术语。

输出格式（严格按此格式）：
---QUERY---
改写后的查询语句
---KEYWORDS---
关键词1 关键词2 关键词3"""

_SIMPLE_USER = "原始问题：{question}"


class SimpleQueryRewriter(QueryRewriter):
    """LLM rewrites one query + extracts keywords."""

    def __init__(self, generate_fn) -> None:
        self._generate = generate_fn
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", _SIMPLE_PROMPT), ("user", _SIMPLE_USER)]
        )

    def rewrite(self, question: str) -> RewriteResult:
        try:
            llm_input = self._prompt.invoke({"question": question})
            messages: list[ChatMessage] = [
                {"role": _LC_TYPE_TO_ROLE.get(msg.type, msg.type), "content": msg.content}
                for msg in llm_input.messages
            ]
            raw = self._generate(messages)
            return self._parse(raw, question)
        except Exception:
            logger.exception("Simple rewrite 失败，回退到原始查询")
            return RewriteResult(queries=[question], keywords=[question])

    @staticmethod
    def _parse(raw: str, fallback: str) -> RewriteResult:
        query = _extract_section(raw, "QUERY") or fallback
        kw_raw = _extract_section(raw, "KEYWORDS") or ""
        keywords = _split_keywords(kw_raw) if kw_raw else [fallback]
        return RewriteResult(queries=[query.strip()], keywords=keywords)


# ---------------------------------------------------------------------------
# Multi-Query rewriter
# ---------------------------------------------------------------------------

_MULTI_PROMPT = """你是一个查询扩展助手。给定一个用户问题，完成两项任务：

1. 生成 {num_queries} 个不同角度的查询变体，用于向量语义检索
2. 提取核心关键词，用于 BM25 关键词检索

规则：
- 每个查询变体从不同角度或措辞表达原始问题
- 关键词只提取内容实义词（名词、术语、实体），去掉虚词和疑问词
- 严格按以下格式输出

输出格式：
---QUERIES---
查询变体1
查询变体2
查询变体3
---KEYWORDS---
关键词1 关键词2 关键词3"""

_MULTI_USER = "原始问题：{question}"


class MultiQueryRewriter(QueryRewriter):
    """LLM generates multiple query variants + extracts keywords."""

    def __init__(self, generate_fn, *, num_queries: int = 3) -> None:
        self._generate = generate_fn
        self._num_queries = num_queries
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", _MULTI_PROMPT), ("user", _MULTI_USER)]
        )

    def rewrite(self, question: str) -> RewriteResult:
        try:
            llm_input = self._prompt.invoke(
                {"question": question, "num_queries": self._num_queries}
            )
            messages: list[ChatMessage] = [
                {"role": _LC_TYPE_TO_ROLE.get(msg.type, msg.type), "content": msg.content}
                for msg in llm_input.messages
            ]
            raw = self._generate(messages)
            return self._parse(raw, question)
        except Exception:
            logger.exception("Multi rewrite 失败，回退到原始查询")
            return RewriteResult(queries=[question], keywords=[question])

    @staticmethod
    def _parse(raw: str, fallback: str) -> RewriteResult:
        queries_block = _extract_section(raw, "QUERIES")
        kw_raw = _extract_section(raw, "KEYWORDS") or ""
        keywords = _split_keywords(kw_raw) if kw_raw else [fallback]

        if queries_block:
            queries = _parse_query_lines(queries_block)
            if queries:
                return RewriteResult(queries=queries, keywords=keywords)
        logger.warning("Multi-Query 解析失败，回退到原始查询")
        return RewriteResult(queries=[fallback], keywords=[fallback])


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


def _extract_section(text: str, tag: str) -> str | None:
    """Extract content between `---TAG---` and the next `---` or end of string."""
    pattern = rf"---\s*{tag}\s*---\s*\n?(.*?)(?=---\s*\w+\s*---|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _parse_query_lines(block: str) -> list[str]:
    """Parse line-delimited queries, stripping numbering/bullets."""
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        line = line.lstrip("•*- ").strip()
        line = re.sub(r"^\d+[\.\)、]\s*", "", line).strip()
        if line:
            cleaned.append(line)
    return cleaned


def _split_keywords(raw: str) -> list[str]:
    """Split keyword string on whitespace / commas into a deduplicated token list."""
    tokens = re.split(r"[\s,，、]+", raw.strip())
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        t = t.strip().lower()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result
