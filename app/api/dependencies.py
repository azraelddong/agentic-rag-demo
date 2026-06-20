from functools import lru_cache

from app.core.config import get_settings
from app.llm.chat_model import OpenAICompatibleChatModel
from app.llm.embedding_model import BgeM3EmbeddingModel, OpenAICompatibleEmbeddingModel
from app.rag.document_loader import DocumentLoader
from app.rag.prompt_builder import PromptBuilder
from app.rag.rag_chain import RAGChain
from app.rag.reranker import NoopReranker
from app.rag.retriever import Retriever
from app.rag.text_splitter import ChunkSplitter
from app.rag.vector_store import MilvusVectorStore
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService


@lru_cache
def get_vector_store() -> MilvusVectorStore:
    return MilvusVectorStore(get_settings())


@lru_cache
def get_embedding_model():
    settings = get_settings()
    if settings.embedding_provider == "bge_m3":
        return BgeM3EmbeddingModel()
    return OpenAICompatibleEmbeddingModel(
        base_url=settings.effective_embedding_base_url,
        api_key=settings.effective_embedding_api_key,
        model_name=settings.embedding_model_name,
    )


@lru_cache
def get_chat_model() -> OpenAICompatibleChatModel:
    settings = get_settings()
    return OpenAICompatibleChatModel(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model_name=settings.llm_model_name,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


@lru_cache
def get_document_service() -> DocumentService:
    settings = get_settings()
    return DocumentService(
        settings=settings,
        loader=DocumentLoader(settings.allowed_document_extensions),
        splitter=ChunkSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embedding_model=get_embedding_model(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_chat_service() -> ChatService:
    settings = get_settings()
    retriever = Retriever(get_embedding_model(), get_vector_store())
    rag_chain = RAGChain(
        settings=settings,
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        chat_model=get_chat_model(),
        reranker=NoopReranker(),
    )
    return ChatService(rag_chain)
