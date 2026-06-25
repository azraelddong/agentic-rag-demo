from __future__ import annotations

from abc import ABC, abstractmethod
import logging

from langchain_core.prompts import ChatPromptTemplate

from app.llm.chat_model import ChatMessage

logger = logging.getLogger(__name__)

# LangChain message .type → OpenAI-compatible role
_LC_TYPE_TO_ROLE: dict[str, str] = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
}

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class QueryRewriter(ABC):
    """Abstract base for query rewriting strategies.

    Each concrete implementation returns a list of search queries derived
    from the original question.  The RAG chain is responsible for running
    retrievals with each query and merging the results.
    """

    @abstractmethod
    def rewrite(self, question: str) -> list[str]:
        """Return one or more search queries. Never returns an empty list."""


# ---------------------------------------------------------------------------
# Simple (single-query) rewriter — current behaviour
# ---------------------------------------------------------------------------

SIMPLE_REWRITE_SYSTEM_PROMPT = """你是一个查询优化助手。你的任务是将用户原始问题改写为更适合知识库向量检索的查询语句。

改写规则：
1. 补充上下文和关键实体，使查询更具体、更聚焦
2. 使用知识库文档中可能出现的术语和关键词
3. 保持原问题的核心意图不变
4. 只输出改写后的问题文本，不要添加任何解释、引号或前缀"""


class SimpleQueryRewriter(QueryRewriter):
    """Use the LLM to rewrite a single question into a retrieval-friendly form."""

    def __init__(self, generate_fn) -> None:
        self._generate = generate_fn
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SIMPLE_REWRITE_SYSTEM_PROMPT),
                ("user", "原始问题：{question}\n改写后的问题："),
            ]
        )

    def rewrite(self, question: str) -> list[str]:
        """Return the rewritten query as a single-element list."""
        try:
            llm_input = self._prompt.invoke({"question": question})
            messages: list[ChatMessage] = [
                {"role": _LC_TYPE_TO_ROLE.get(msg.type, msg.type), "content": msg.content}
                for msg in llm_input.messages
            ]
            rewritten = self._generate(messages)
            if rewritten and rewritten != question:
                return [rewritten]
            logger.warning("Query rewrite 返回相同或空文本，保持原始查询")
            return [question]
        except Exception:
            logger.exception("Query rewrite 失败，回退到原始查询")
            return [question]


# ---------------------------------------------------------------------------
# Multi-Query rewriter — generates N diverse queries
# ---------------------------------------------------------------------------

MULTI_QUERY_SYSTEM_PROMPT = """你是一个查询扩展助手。给定一个用户问题，生成 {num_queries} 个不同角度的查询变体，用于从知识库中检索相关文档。

要求：
1. 每个查询变体从不同角度或使用不同措辞表达原始问题
2. 使用知识库文档中可能出现的术语和同义词
3. 保持原始问题的核心意图，但可以侧重不同方面
4. 每行输出一个查询，不要编号、引号或解释
5. 严格输出 {num_queries} 行，不要多也不要少"""


class MultiQueryRewriter(QueryRewriter):
    """Use the LLM to generate multiple diverse query variants."""

    def __init__(self, generate_fn, *, num_queries: int = 3) -> None:
        self._generate = generate_fn
        self._num_queries = num_queries
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MULTI_QUERY_SYSTEM_PROMPT),
                ("user", "原始问题：{question}\n{num_queries} 个查询变体："),
            ]
        )

    def rewrite(self, question: str) -> list[str]:
        """Return multiple query variants, or [original] on failure."""
        try:
            llm_input = self._prompt.invoke(
                {"question": question, "num_queries": self._num_queries}
            )
            messages: list[ChatMessage] = [
                {"role": _LC_TYPE_TO_ROLE.get(msg.type, msg.type), "content": msg.content}
                for msg in llm_input.messages
            ]
            raw = self._generate(messages)
            queries = self._parse_queries(raw)
            if queries:
                return queries
            logger.warning("Multi-Query 解析失败，回退到原始查询")
            return [question]
        except Exception:
            logger.exception("Multi-Query 生成失败，回退到原始查询")
            return [question]

    def _parse_queries(self, raw: str) -> list[str]:
        """Parse line-delimited queries from the LLM response."""
        raw = raw.strip()
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        if not lines:
            return []

        # Remove common numbering prefixes like "1." / "1)" / "- "
        cleaned: list[str] = []
        for line in lines:
            line = line.lstrip("•*- ").strip()
            # Strip "1. " or "1) " or "1、" patterns
            import re
            line = re.sub(r"^\d+[\.\)、]\s*", "", line).strip()
            if line:
                cleaned.append(line)

        return (cleaned or lines)[: self._num_queries]
