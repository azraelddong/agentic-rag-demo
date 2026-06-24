from typing import Any

from pydantic import BaseModel, Field

"""问答请求参数"""
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索召回数量")
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Milvus metadata filter, e.g. {'source_type': 'md'}",
    )

"""来源结果"""
class Source(BaseModel):
    file_name: str | None = None
    file_path: str | None = None
    chunk_index: int | None = None
    source_type: str | None = None
    score: float


"""问题响应结果"""
class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
