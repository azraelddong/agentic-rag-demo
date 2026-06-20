import logging
from typing import Any

from app.core.config import Settings
from app.llm.chat_model import ChatModel
from app.rag.prompt_builder import PromptBuilder
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = "知识库中未找到相关信息"


class RAGChain:
    """Basic retrieval-augmented generation chain for phase 1."""

    def __init__(
        self,
        *,
        settings: Settings,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        chat_model: ChatModel,
        reranker: Reranker,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.chat_model = chat_model
        self.reranker = reranker

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> tuple[str, list[SearchResult]]:
        normalized_question = question.strip()
        results = self.retriever.retrieve(
            normalized_question,
            top_k=top_k or self.settings.rag_top_k,
            metadata_filter=metadata_filter,
        )
        results = self._filter_by_score(results)
        results = self.reranker.rerank(normalized_question, results)

        if not results:
            logger.info("No relevant chunks found for query")
            return NOT_FOUND_ANSWER, []

        messages = self.prompt_builder.build_messages(normalized_question, results)
        answer = self.chat_model.generate(messages)
        if not answer:
            return NOT_FOUND_ANSWER, results
        return answer, results

    def _filter_by_score(self, results: list[SearchResult]) -> list[SearchResult]:
        threshold = self.settings.rag_score_threshold
        if threshold is None:
            return results

        metric_type = self.settings.milvus_metric_type.upper()
        if metric_type == "L2":
            return [result for result in results if result.score <= threshold]
        return [result for result in results if result.score >= threshold]
