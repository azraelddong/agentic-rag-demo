from abc import ABC, abstractmethod
import logging

from openai import OpenAI, OpenAIError

from app.core.exceptions import ConfigurationError, LLMProviderError

logger = logging.getLogger(__name__)

"""抽象的Embedding模型接口，定义了文本向量化的方法。"""
class EmbeddingModel(ABC):
    """Embedding model interface for remote and local embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

"""Embedding模型接口，支持远程和本地的Embedding提供者。定义了一个抽象方法embed_texts用于批量文本的向量化，以及一个默认实现的embed_query方法用于单个查询的向量化。"""
class OpenAICompatibleEmbeddingModel(EmbeddingModel):
    """Embedding adapter for OpenAI-compatible embedding APIs."""

    def __init__(self, *, base_url: str, api_key: str, model_name: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ConfigurationError("Embedding API Key 未配置，请先在 .env 中填写 API Key")
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            return []

        try:
            response = self._get_client().embeddings.create(
                model=self.model_name,
                input=clean_texts,
            )
            return [item.embedding for item in response.data]
        except ConfigurationError:
            raise
        except OpenAIError as exc:
            logger.exception("Embedding provider request failed")
            raise LLMProviderError("Embedding 调用失败", detail={"error": str(exc)}) from exc

"""基于bge-m3的Embedding模型，目前作为本地Embedding的预留扩展点，尚未实现具体的向量化逻辑。"""
class BgeM3EmbeddingModel(EmbeddingModel):
    """Reserved extension point for local bge-m3 embeddings."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "bge-m3 本地 Embedding 预留：可在此接入 sentence-transformers 或推理服务"
        )
