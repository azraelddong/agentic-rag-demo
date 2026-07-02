"""结构化记忆条目数据模型。

每条记忆条目都带有可信度、来源追踪和生命周期状态字段，
遵循项目记忆 [[agent-memory-pitfalls]] 的避坑原则。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------

class MemoryEntryType(str, Enum):
    """记忆条目类型 —— 对应"应记住"的 6 种场景。

    分类器也可以返回 ``"discard"`` 表示不存储，但 discard 不作为
    MemoryEntryType 枚举值对外暴露，仅在分类器内部使用。
    """

    preference = "preference"            # 用户明确表达的长期偏好
    work_habit = "work_habit"             # 用户反复出现的工作习惯 / 模式
    business_config = "business_config"   # 稳定的业务系统配置
    experience = "experience"             # 有效的任务执行经验
    correction = "correction"             # 用户明确纠正的错误
    fix_strategy = "fix_strategy"         # Agent 工具调用失败后的修正策略


class MemoryEntryStatus(str, Enum):
    """记忆条目生命周期状态。

    active → conflicted / pending_review → archived / expired
    """

    active = "active"                    # 正常生效中
    pending_review = "pending_review"    # 待人工审核
    conflicted = "conflicted"            # 存在冲突，等待解决
    archived = "archived"                # 已被新版本替代（保留用于溯源）
    expired = "expired"                  # 已过期


# 分类器内部使用的判别结果
_DISCARD = "discard"
"""内部标记：不存储。不暴露为 MemoryEntryType。"""

ALL_MEMORY_ENTRY_TYPES: frozenset[str] = frozenset(
    t.value for t in MemoryEntryType
)
ALL_CLASSIFIER_OUTCOMES: frozenset[str] = ALL_MEMORY_ENTRY_TYPES | {_DISCARD}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TurnToolInfo:
    """本轮对话中发生的工具调用摘要。

    由调用方（AgentService / demo agent）从消息列表中提取，
    传入 Gatekeeper.process_turn() 作为分类依据。
    """

    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""            # 工具返回摘要，截断至 500 字符
    success: bool = True
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Pydantic 模型
# ---------------------------------------------------------------------------

class MemoryCandidate(BaseModel):
    """LLM 分类器输出的单条候选（尚未持久化）。

    entry_type 为 ``"discard"`` 的候选在 gatekeeper 中被丢弃，
    不会进一步持久化为 MemoryEntry。
    """

    entry_type: str = Field(
        ...,
        description=f"记忆类型: {' | '.join(sorted(ALL_MEMORY_ENTRY_TYPES))} 或 discard",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="记忆条目正文",
    )
    summary: str = Field(
        default="",
        max_length=200,
        description="一句话摘要，用于快速预览和去重匹配",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM 判定的可信度 0.0~1.0",
    )
    reason: str = Field(
        default="",
        description="LLM 给出的分类理由（可解释性）",
    )

    @field_validator("entry_type")
    @classmethod
    def _check_entry_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ALL_CLASSIFIER_OUTCOMES:
            raise ValueError(
                f"非法 entry_type '{v}', 允许: {sorted(ALL_CLASSIFIER_OUTCOMES)}"
            )
        return v

    @property
    def should_discard(self) -> bool:
        """该候选是否应被丢弃（不持久化）。"""
        return self.entry_type == _DISCARD

    @property
    def memory_entry_type(self) -> MemoryEntryType | None:
        """返回对应的 MemoryEntryType 枚举，discard 时返回 None。"""
        if self.entry_type == _DISCARD:
            return None
        return MemoryEntryType(self.entry_type)


class MemoryEntry(BaseModel):
    """结构化记忆条目 —— 持久化在 Redis mem:entry 命名空间。

    与原始消息窗口 (ConversationMemory) 完全解耦：
    - 原始消息 = 完整对话窗口（TTL 1 小时，用于上下文）
    - 记忆条目 = 提炼后的结构化信息（TTL 30 天，用于长期记忆）
    """

    entry_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="唯一标识，使用 UUID hex 实现幂等写入",
    )
    entry_type: MemoryEntryType = Field(
        ...,
        description="记忆条目类型",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="记忆条目的核心内容",
    )
    summary: str = Field(
        default="",
        max_length=200,
        description="一句话摘要",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="可信度 0.0~1.0。用户直接陈述为 1.0，LLM 推断降低",
    )

    # ---- 来源追踪 ----
    source_session_id: str = Field(
        default="",
        description="产生该记忆的会话 ID",
    )
    source_turn: int = Field(
        default=0,
        ge=0,
        description="会话中的第几轮对话",
    )

    # ---- 生命周期 ----
    status: MemoryEntryStatus = Field(
        default=MemoryEntryStatus.active,
        description="生命周期状态",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="数据格式版本，每次更新 +1",
    )
    conflict_group: str | None = Field(
        default=None,
        description="冲突组 ID。同一组内条目语义相近，需人工或自动裁决",
    )

    # ---- 时间戳 ----
    created_at: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(),
        description="ISO 8601 创建时间",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(),
        description="ISO 8601 最后更新时间",
    )
    expires_at: str | None = Field(
        default=None,
        description="显式过期时间。None 表示由 Redis TTL 控制",
    )

    # ---- 人工审核 ----
    reviewed_by: str | None = Field(
        default=None,
        description="审核人标识",
    )
    reviewed_at: str | None = Field(
        default=None,
        description="审核时间 ISO 8601",
    )

    # ---- 扩展 ----
    tags: list[str] = Field(
        default_factory=list,
        description="标签列表，方便检索",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据，如 mention_count、source_tool 等",
    )

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    @classmethod
    def from_candidate(
        cls,
        candidate: MemoryCandidate,
        *,
        source_session_id: str = "",
        source_turn: int = 0,
    ) -> MemoryEntry:
        """从分类器候选创建 MemoryEntry。

        仅当 candidate.entry_type 合法且非 discard 时调用。
        """
        entry_type = candidate.memory_entry_type
        if entry_type is None:
            raise ValueError(f"不能从 discard 候选创建 MemoryEntry: {candidate.summary}")
        now = datetime.now(tz=timezone.utc).isoformat()
        return cls(
            entry_type=entry_type,
            content=candidate.content,
            summary=candidate.summary,
            confidence=candidate.confidence,
            source_session_id=source_session_id,
            source_turn=source_turn,
            created_at=now,
            updated_at=now,
        )

    def bump_version(self) -> None:
        """递增版本号并更新 updated_at。"""
        self.version += 1
        self.updated_at = datetime.now(tz=timezone.utc).isoformat()

    def mark_status(self, new_status: MemoryEntryStatus) -> None:
        """更新状态并刷新 updated_at。"""
        self.status = new_status
        self.updated_at = datetime.now(tz=timezone.utc).isoformat()

    def archive(self) -> None:
        """归档（被新版本替代时调用）。"""
        self.mark_status(MemoryEntryStatus.archived)
