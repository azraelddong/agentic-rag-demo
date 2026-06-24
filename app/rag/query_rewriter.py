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

REWRITE_SYSTEM_PROMPT = """你是一个查询优化助手。你的任务是将用户原始问题改写为更适合知识库向量检索的查询语句。

改写规则：
1. 补充上下文和关键实体，使查询更具体、更聚焦
2. 使用知识库文档中可能出现的术语和关键词
3. 保持原问题的核心意图不变
4. 只输出改写后的问题文本，不要添加任何解释、引号或前缀"""


class QueryRewriter:
    """Use the LLM to rewrite a user question into a retrieval-friendly form."""

    def __init__(self, generate_fn) -> None:
        self._generate = generate_fn
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REWRITE_SYSTEM_PROMPT),
                ("user", "原始问题：{question}\n改写后的问题："),
            ]
        )

    def rewrite(self, question: str) -> str:
        """Return the rewritten query, or the original on failure."""
        try:
            llm_input = self._prompt.invoke({"question": question})
            messages: list[ChatMessage] = [
                {"role": _LC_TYPE_TO_ROLE.get(msg.type, msg.type), "content": msg.content}
                for msg in llm_input.messages
            ]
            rewritten = self._generate(messages)
            if rewritten and rewritten != question:
                return rewritten
            logger.warning("Query rewrite 返回相同或空文本，保持原始查询")
            return question
        except Exception:
            logger.exception("Query rewrite 失败，回退到原始查询")
            return question
