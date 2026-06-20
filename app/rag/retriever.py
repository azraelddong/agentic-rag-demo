from typing import Any

from app.llm.embedding_model import EmbeddingModel
from app.rag.vector_store import MilvusVectorStore, SearchResult


class Retriever:
    """Embed the user query and retrieve similar chunks from Milvus."""

    def __init__(self, embedding_model: EmbeddingModel, vector_store: MilvusVectorStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_vector = self.embedding_model.embed_query(query)
        return self.vector_store.similarity_search(
            query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
