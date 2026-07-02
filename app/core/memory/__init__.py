"""短期会话记忆模块。

提供 Redis 支持的会话记忆，demo agent 和 app agent 均可复用。

子模块：
- session_store:       Redis 底层 KV 封装
- conversation_memory: 原始消息窗口（多轮上下文）
- gatekeeper:          结构化记忆准入守卫（分类 + 过滤 + 冲突检测）
- entry_models:        结构化记忆条目数据模型
- classifier:          LLM 记忆分类器
- models:              会话元数据模型
"""

from app.core.memory.classifier import MemoryClassifier
from app.core.memory.conversation_memory import ConversationMemory
from app.core.memory.entry_models import (
    MemoryCandidate,
    MemoryEntry,
    MemoryEntryStatus,
    MemoryEntryType,
    TurnToolInfo,
)
from app.core.memory.gatekeeper import MemoryGatekeeper, rule_based_filter
from app.core.memory.models import SessionMetadata
from app.core.memory.session_store import RedisSessionStore

__all__ = [
    # 原始消息存储
    "ConversationMemory",
    "RedisSessionStore",
    "SessionMetadata",
    # 结构化记忆
    "MemoryEntry",
    "MemoryEntryType",
    "MemoryEntryStatus",
    "MemoryCandidate",
    "TurnToolInfo",
    "MemoryClassifier",
    "MemoryGatekeeper",
    "rule_based_filter",
]
