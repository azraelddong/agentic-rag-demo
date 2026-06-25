from functools import lru_cache

from app.core.config import get_settings
from app.llm.chat_model import OpenAICompatibleChatModel
from app.llm.embedding_model import BgeM3EmbeddingModel, OpenAICompatibleEmbeddingModel
from app.rag.document_loader import DocumentLoader
from app.rag.prompt_builder import PromptBuilder
from app.rag.rag_chain import RAGChain
from app.rag.bm25_retriever import BM25Retriever
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_rewriter import MultiQueryRewriter, QueryRewriter, SimpleQueryRewriter
from app.rag.reranker import BgeReranker, JinaReranker, NoopReranker, SiliconFlowReranker
from app.rag.retriever import Retriever
from app.rag.text_splitter import ChunkSplitter
from app.rag.vector_store import MilvusVectorStore
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService


"""依赖注入模块，提供应用中各个组件的单例实例。使用lru_cache装饰器实现简单的单例模式，确保每个组件在应用生命周期内只创建一次。"""

"""获取MilvusVectorStore实例，连接到Milvus向量数据库。"""
@lru_cache
def get_vector_store() -> MilvusVectorStore:
    return MilvusVectorStore(get_settings())

"""获取EmbeddingModel嵌入实例，根据配置选择具体的实现。"""
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

"""获取ChatModel实例，根据配置创建OpenAI兼容的聊天模型。"""
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

"""获取Reranker实例，根据rerank_provider配置选择具体的实现。"""
@lru_cache
def get_reranker():
    settings = get_settings()
    provider = settings.rerank_provider
    if provider == "siliconflow":
        return SiliconFlowReranker(
            base_url=settings.effective_rerank_base_url,
            api_key=settings.effective_rerank_api_key,
            model_name=settings.rerank_model_name,
        )
    if provider == "jina":
        return JinaReranker(
            base_url=settings.effective_rerank_base_url,
            api_key=settings.effective_rerank_api_key,
            model_name=settings.rerank_model_name,
        )
    if provider == "bge":
        return BgeReranker()
    return NoopReranker()

"""获取QueryRewriter实例，根据query_rewrite_method配置选择改写策略。"""
@lru_cache
def get_query_rewriter() -> QueryRewriter | None:
    settings = get_settings()
    method = settings.query_rewrite_method
    if method == "simple":
        return SimpleQueryRewriter(get_chat_model().generate)
    if method == "multi":
        return MultiQueryRewriter(
            get_chat_model().generate,
            num_queries=settings.query_rewrite_multi_count,
        )
    return None

"""获取DocumentService实例，注入所需的依赖组件。"""
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

"""获取BM25Retriever实例（keyword检索），被HybridRetriever复用。"""
@lru_cache
def get_bm25_retriever() -> BM25Retriever:
    return BM25Retriever(get_vector_store())

"""获取Retriever实例，根据retrieval_method配置选择检索策略。"""
@lru_cache
def get_retriever() -> Retriever | HybridRetriever:
    settings = get_settings()
    if settings.retrieval_method == "hybrid":
        return HybridRetriever(
            dense_retriever=Retriever(get_embedding_model(), get_vector_store()),
            bm25_retriever=get_bm25_retriever(),
        )
    return Retriever(get_embedding_model(), get_vector_store())

"""获取ChatService实例，注入所需的依赖组件，包括RAGChain和相关的子组件。"""
@lru_cache
def get_chat_service() -> ChatService:
    settings = get_settings()
    rag_chain = RAGChain(
        settings=settings,
        retriever=get_retriever(),
        prompt_builder=PromptBuilder(),
        chat_model=get_chat_model(),
        reranker=get_reranker(),
        query_rewriter=get_query_rewriter(),
    )
    return ChatService(rag_chain)
