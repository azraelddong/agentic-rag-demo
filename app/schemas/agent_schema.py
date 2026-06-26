from typing import Any

from pydantic import BaseModel, Field

from app.schemas.chat_schema import Source


class AgentRequest(BaseModel):
    """Agent ask request – fields kept consistent with ChatRequest."""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索召回数量")
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Milvus metadata filter, e.g. {'source_type': 'md'}",
    )


class AgentTraceStep(BaseModel):
    """One visible step in the Agent graph execution trace."""

    node: str = Field(..., description="执行节点名称")
    decision: str | None = Field(default=None, description="节点决策")
    reflection: str | None = Field(default=None, description="反思结果")
    source_count: int | None = Field(default=None, description="来源数量")
    top_k: int | None = Field(default=None, description="本轮检索 top_k")
    retry_reason: str | None = Field(default=None, description="重试原因")


class AgentTrace(BaseModel):
    """Lightweight trace for Agent graph execution."""

    plan: str = Field(default="rag_search", description="规划动作")
    reflection: str = Field(
        default="supported",
        description="反思结果: supported 或 insufficient_context",
    )
    iterations: int = Field(default=1, description="执行轮数")
    steps: list[AgentTraceStep] = Field(
        default_factory=list,
        description="Agent graph execution steps",
    )


class AgentResponse(BaseModel):
    """Agent ask response – extends ChatResponse with an execution trace."""

    answer: str
    sources: list[Source]
    trace: AgentTrace
