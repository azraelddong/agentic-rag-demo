from collections.abc import Sequence

from app.llm.chat_model import ChatMessage
from app.rag.vector_store import SearchResult

"""提示词构建器，将检索到的上下文和用户问题构建成符合LLM输入格式的消息列表。"""
class PromptBuilder:
    """Build a strict RAG prompt grounded in retrieved context."""

    """系统提示词"""
    system_prompt = (
        "你是企业知识库问答助手。必须只基于提供的检索上下文回答问题。"
        "如果上下文中没有相关信息，必须回答：知识库中未找到相关信息。"
        "不要编造上下文以外的事实。"
    )

    """构建消息列表"""
    def build_messages(self, question: str, contexts: Sequence[SearchResult]) -> list[ChatMessage]:
        context_text = self._format_contexts(contexts)
        user_prompt = (
            f"用户问题：\n{question}\n\n"
            f"检索上下文：\n{context_text}\n\n"
            "请给出简洁、准确的答案，并在答案中尽量体现来源依据。"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_contexts(self, contexts: Sequence[SearchResult]) -> str:
        blocks: list[str] = []
        for index, result in enumerate(contexts, start=1):
            metadata = result.metadata
            blocks.append(
                "\n".join(
                    [
                        f"[context_{index}]",
                        f"file_name: {metadata.get('file_name')}",
                        f"chunk_index: {metadata.get('chunk_index')}",
                        f"score: {result.score:.4f}",
                        result.text,
                    ]
                )
            )
        return "\n\n".join(blocks)
