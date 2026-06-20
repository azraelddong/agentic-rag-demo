from abc import ABC, abstractmethod
from collections.abc import Sequence
import logging

from openai import OpenAI, OpenAIError

from app.core.exceptions import ConfigurationError, LLMProviderError

logger = logging.getLogger(__name__)

ChatMessage = dict[str, str]


class ChatModel(ABC):
    """Chat completion interface for OpenAI-compatible providers."""

    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage]) -> str:
        """Generate an answer from chat messages."""


class OpenAICompatibleChatModel(ChatModel):
    """Adapter for OpenAI, Qwen, DeepSeek, or any compatible chat API."""

    def __init__(
            self,
            *,
            base_url: str,
            api_key: str,
            model_name: str,
            temperature: float,
            max_tokens: int,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ConfigurationError("LLM_API_KEY 未配置，请先在 .env 中填写 API Key")
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=list(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except ConfigurationError:
            raise
        except OpenAIError as exc:
            logger.exception("LLM provider request failed")
            raise LLMProviderError("LLM 调用失败", detail={"error": str(exc)}) from exc
