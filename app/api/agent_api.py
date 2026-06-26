from fastapi import APIRouter, Depends

from app.api.dependencies import get_agent_service
from app.agent.agent_service import AgentService
from app.core.config import get_settings
from app.core.exceptions import DocumentProcessingError
from app.schemas.agent_schema import AgentRequest, AgentResponse

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/ask", response_model=AgentResponse)
def ask(
    request: AgentRequest,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentResponse:
    max_chars = get_settings().query_max_chars

    if len(request.question.strip()) > max_chars:
        raise DocumentProcessingError(
            "问题长度超过限制",
            detail={"max_chars": max_chars},
        )

    return agent_service.ask(
        question=request.question,
        top_k=request.top_k,
        metadata_filter=request.metadata_filter,
    )
