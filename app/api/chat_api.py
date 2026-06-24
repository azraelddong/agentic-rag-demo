from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.core.config import get_settings
from app.core.exceptions import DocumentProcessingError
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["chat"])

"""问题接口，接收用户问题并返回回答。依赖注入ChatService来处理业务逻辑。"""
@router.post("/ask", response_model=ChatResponse)
def ask(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),  # 依赖注入 ChatService
) -> ChatResponse:
    max_chars = get_settings().query_max_chars  # 从配置中获取最大字符数限制

    """验证问题长度，超过限制则抛出异常"""
    if len(request.question.strip()) > max_chars:
        raise DocumentProcessingError(
            "问题长度超过限制",
            detail={"max_chars": max_chars},
        )

    return chat_service.ask(
        question=request.question,
        top_k=request.top_k,
        metadata_filter=request.metadata_filter,
    )
