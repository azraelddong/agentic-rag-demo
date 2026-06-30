from functools import lru_cache
import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""从.env文件和环境变量加载的运行时配置。"""
class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and .env."""

    """BaseSettings内部自动读取Settings.model_config来配置环境变量加载行为。"""
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore", # 忽略未在Settings中定义的环境变量，避免因环境中存在无关变量而导致加载失败
    )

    app_name: str = "agentic-rag-demo"
    app_env: str = "local"
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_retention_days: int = 180
    log_max_file_size_mb: int = 20
    docs_dir: str = "docs"
    upload_max_mb: int = 20
    allowed_document_extensions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["txt", "md", "pdf"]
    )

    chunk_size: int = 800
    chunk_overlap: int = 100

    """LLM相关配置，支持OpenAI兼容接口。"""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    """Embedding相关配置，支持OpenAI兼容接口。"""
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 64
    embedding_provider: str = "openai_compatible"

    """Reranking相关配置"""
    rerank_provider: str = "none"
    rerank_model_name: str = "bge-reranker-v2-m3"
    rerank_base_url: str | None = None
    rerank_api_key: str | None = None
    rerank_top_n: int = 5
    rerank_retrieval_k: int = 20

    """milvus相关配置"""
    milvus_uri: str = "http://localhost:19530"
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_no_grpc_proxy: str = ""
    milvus_collection_name: str = "agentic_rag_chunks"
    milvus_metric_type: str = "COSINE"
    milvus_index_type: str = "AUTOINDEX"
    milvus_consistency_level: str = "Strong"

    rag_top_k: int = 5
    rag_score_threshold: float | None = 0.2
    query_max_chars: int = 2000
    retrieval_method: str = "dense"
    query_rewrite_method: str = "none"
    query_rewrite_multi_count: int = 3

    """Redis / short-term memory settings"""
    redis_url: str = "redis://localhost:6379/0"
    redis_session_ttl: int = Field(
        default=3600,
        description="TTL in seconds for session memory keys",
        ge=60,
        le=86400,
    )
    redis_session_prefix: str = "mem:session"

    @field_validator("allowed_document_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                return [item.strip().lstrip(".").lower() for item in json.loads(raw)]
            return [item.strip().lstrip(".").lower() for item in raw.split(",") if item]
        return value

    @field_validator("milvus_metric_type", "milvus_index_type", mode="before")
    @classmethod
    def normalize_upper(cls, value: str) -> str:
        return value.upper()

    @field_validator("embedding_provider", "rerank_provider", "retrieval_method", "query_rewrite_method", mode="before")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.strip().lower().replace("-", "_")

    @field_validator("rerank_top_n", "rerank_retrieval_k", mode="before")
    @classmethod
    def strip_rerank_integer_inline_comment(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.split("#", 1)[0].strip()
        return value

    @property
    def docs_path(self) -> Path:
        path = Path(self.docs_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def log_path(self) -> Path:
        path = Path(self.log_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path / f"{self.app_name}.log"

    @property
    def max_upload_bytes(self) -> int:
        return self.upload_max_mb * 1024 * 1024

    @property
    def log_max_file_size_bytes(self) -> int:
        return self.log_max_file_size_mb * 1024 * 1024

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key

    @property
    def effective_rerank_base_url(self) -> str:
        if self.rerank_base_url:
            return self.rerank_base_url
        provider_defaults = {
            "siliconflow": "https://api.siliconflow.cn/v1",
            "jina": "https://api.jina.ai/v1",
        }
        return provider_defaults.get(self.rerank_provider, self.llm_base_url)

    @property
    def effective_rerank_api_key(self) -> str:
        return self.rerank_api_key or self.llm_api_key

    @property
    def rerank_enabled(self) -> bool:
        return self.rerank_provider != "none"

    @property
    def query_rewrite_enabled(self) -> bool:
        return self.query_rewrite_method != "none"


@lru_cache
def get_settings() -> Settings:
    return Settings()
