from http import HTTPStatus

"""自定义异常类"""
class AppError(Exception):
    """Base application error surfaced through FastAPI handlers."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class ConfigurationError(AppError):
    status_code = HTTPStatus.BAD_REQUEST
    code = "configuration_error"


class DocumentProcessingError(AppError):
    status_code = HTTPStatus.BAD_REQUEST
    code = "document_processing_error"


class VectorStoreError(AppError):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "vector_store_error"


class LLMProviderError(AppError):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "llm_provider_error"


class RAGPipelineError(AppError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "rag_pipeline_error"
