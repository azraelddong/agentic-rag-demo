from typing import Any

from app.rag.rag_chain import RAGChain
from app.rag.vector_store import SearchResult
from app.schemas.chat_schema import ChatResponse, Source


class ChatService:
    """Chat application service wrapping the RAG chain."""

    def __init__(self, rag_chain: RAGChain) -> None:
        self.rag_chain = rag_chain

    def ask(
        self,
        *,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> ChatResponse:
        answer, results = self.rag_chain.ask(
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return ChatResponse(answer=answer, sources=self._build_sources(results))

    def _build_sources(self, results: list[SearchResult]) -> list[Source]:
        sources: list[Source] = []
        for result in results:
            metadata = result.metadata
            sources.append(
                Source(
                    file_name=metadata.get("file_name"),
                    file_path=metadata.get("file_path"),
                    chunk_index=metadata.get("chunk_index"),
                    source_type=metadata.get("source_type"),
                    score=result.score,
                )
            )
        return sources
