"""会话记忆数据模型。

每条记忆都带有可信度字段，避免把低置信度推测当事实（避坑 3）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SessionMetadata(BaseModel):
    """会话记忆的元数据，与消息列表分开存储以便独立查询。

    Attributes:
        source_type: 记忆来源类型，会话记忆固定为 ``"conversation"``。
        confidence: 可信度 0.0~1.0，用户直接输入为 1.0。
        created_at / updated_at: ISO 8601 时间戳。
        expires_at: 过期时间，由 Redis TTL 控制时可为 None。
        version: 数据格式版本，结构变更时递增。
        status: ``"active"`` | ``"expired"`` | ``"cleared"``。
    """

    source_type: str = Field(
        default="conversation",
        description="Memory source type",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score: 1.0 for direct user input",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(),
        description="ISO 8601 last-update timestamp",
    )
    expires_at: str | None = Field(
        default=None,
        description="Expiry timestamp; None means TTL-controlled by Redis",
    )
    version: int = Field(default=1, description="Schema version")
    status: str = Field(default="active", description="active | expired | cleared")
    total_messages: int = Field(default=0, description="Total messages in session")
    total_turns: int = Field(default=0, description="User-assistant turn pairs")
    stored_messages: int = Field(default=0, description="Messages after windowing")
