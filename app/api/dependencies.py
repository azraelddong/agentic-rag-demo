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
from app.agent.agent_service import AgentService
from app.core.memory.conversation_memory import ConversationMemory
from app.core.memory.session_store import RedisSessionStore
from app.core.memory.gatekeeper import MemoryGatekeeper
from app.core.memory.classifier import MemoryClassifier
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

"""获取Dense Retriever实例（纯向量检索）。"""
@lru_cache
def get_dense_retriever() -> Retriever:
    return Retriever(get_embedding_model(), get_vector_store())


"""获取Hybrid Retriever实例（dense + BM25混合检索）。"""
@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        dense_retriever=get_dense_retriever(),
        bm25_retriever=get_bm25_retriever(),
    )


"""获取Retriever实例，根据retrieval_method配置选择检索策略。"""
@lru_cache
def get_retriever() -> Retriever | HybridRetriever:
    settings = get_settings()
    if settings.retrieval_method == "hybrid":
        return get_hybrid_retriever()
    return get_dense_retriever()

"""获取RAGChain实例，作为Basic RAG和Agentic RAG的共享核心链路。"""
@lru_cache
def get_rag_chain() -> RAGChain:
    settings = get_settings()
    return RAGChain(
        settings=settings,
        retriever=get_retriever(),
        prompt_builder=PromptBuilder(),
        chat_model=get_chat_model(),
        reranker=get_reranker(),
        query_rewriter=get_query_rewriter(),
    )


"""获取ChatService实例，复用共享的RAGChain实例。"""
@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(get_rag_chain())


"""获取Agent专用的MultiQueryRewriter，用于retry时的强制query rewrite。"""
@lru_cache
def get_agent_retry_query_rewriter() -> QueryRewriter:
    return MultiQueryRewriter(
        get_chat_model().generate,
        num_queries=get_settings().query_rewrite_multi_count,
    )


"""获取RedisSessionStore单例，所有会话记忆共享同一个 Redis 连接。"""
@lru_cache
def get_session_store() -> RedisSessionStore:
    settings = get_settings()
    return RedisSessionStore(
        redis_url=settings.redis_url,
        password=settings.redis_password,
        default_ttl=settings.redis_session_ttl,
        key_prefix=settings.redis_session_prefix,
    )


"""获取ConversationMemory单例，注入RedisSessionStore。"""
@lru_cache
def get_conversation_memory() -> ConversationMemory:
    return ConversationMemory(store=get_session_store())


"""获取 Gatekeeper 专用 RedisSessionStore 单例（独立 key 前缀和 TTL）。"""
@lru_cache
def get_entry_store() -> RedisSessionStore:
    settings = get_settings()
    return RedisSessionStore(
        redis_url=settings.redis_url,
        password=settings.redis_password,
        default_ttl=settings.redis_entry_ttl,
        key_prefix=settings.redis_entry_prefix,
    )


"""获取 MemoryGatekeeper 单例（gatekeeper_enabled=False 时返回 None）。"""
@lru_cache
def get_memory_gatekeeper() -> MemoryGatekeeper | None:
    settings = get_settings()
    if not settings.gatekeeper_enabled:
        return None
    # 将 ChatModel.generate 包装为 classifier 需要的 (str) -> str 函数
    chat_model = get_chat_model()
    llm_func = lambda prompt: chat_model.generate([
        {"role": "user", "content": prompt}
    ])
    return MemoryGatekeeper(
        store=get_entry_store(),
        classifier=MemoryClassifier(llm_func=llm_func),
    )


"""获取AgentService实例，注入共享RAGChain、retry专用依赖、会话记忆和Gatekeeper。"""
@lru_cache
def get_agent_service() -> AgentService:
    return AgentService(
        rag_chain=get_rag_chain(),
        retry_retriever=get_hybrid_retriever(),
        retry_query_rewriter=get_agent_retry_query_rewriter(),
        memory=get_conversation_memory(),
        gatekeeper=get_memory_gatekeeper(),
    )
