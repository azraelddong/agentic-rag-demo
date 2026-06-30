"""会话记忆 — 消息持久化、窗口裁剪、元数据追踪。

在 agent 循环中的典型用法::

    memory = ConversationMemory(store)
    messages = memory.load_messages(session_id)       # 加载历史
    messages.append(HumanMessage(content=user_input))  # 追加当前消息
    result = agent.invoke({"messages": messages})      # 调用 agent
    memory.save_messages(session_id, result["messages"])  # 持久化

设计原则（来自项目记忆 [[agent-memory-pitfalls]]）：
- 坑 1 避免：不做全文向量检索，只存最近 N 轮窗口
- 坑 2 避免：TTL 自动过期 + 窗口裁剪 + /clear 手动清理
- 坑 3 避免：每条记忆附带 SessionMetadata 可信度字段
- 坑 4 避免：key 前缀 ``mem:session`` 与 Milvus 知识库隔离
- 坑 5 避免：不修改系统提示词，仅管理消息列表
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

from app.core.memory.session_store import RedisSessionStore

logger = logging.getLogger(__name__)

# Redis sub-keys ──────────────────────────────────────────────────────
SUB_KEY_MESSAGES = "messages"
SUB_KEY_METADATA = "metadata"

# 默认窗口大小：保留最近 N 轮（一轮 = 用户问题 + assistant 回答）
DEFAULT_MAX_TURNS = 25


class ConversationMemory:
    """Redis 支持的会话级短期记忆。

    自动处理：
    - 消息序列化/反序列化（通过 LangChain message_to_dict / messages_from_dict）
    - 消息窗口裁剪（保留最近 max_turns 轮）
    - 元数据追踪（创建时间、更新次数、消息数、可信度）
    - TTL 自动续期（每次读写刷新 Redis key 过期时间）

    Args:
        store: 底层 Redis 存储实例。
        max_turns: 保留的最近轮数上限，默认 25 轮（约 50 条消息）。
    """

    def __init__(
        self,
        store: RedisSessionStore,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.store = store
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_messages(self, session_id: str) -> list[BaseMessage]:
        """加载会话的历史消息。

        新会话返回空列表；数据损坏时记录警告并返回空列表。
        """
        raw = self.store.get(session_id, SUB_KEY_MESSAGES)
        if not raw:
            return []
        try:
            data = json.loads(raw)
            msgs = messages_from_dict(data)
            logger.info(
                "MEM LOAD  session=%s  %d messages loaded",
                session_id,
                len(msgs),
            )
            return msgs
        except Exception as exc:
            logger.warning(
                "MEM LOAD  session=%s  deserialize failed, discarding: %s",
                session_id,
                exc,
            )
            return []

    def save_messages(self, session_id: str, messages: list[BaseMessage]) -> None:
        """保存会话消息，自动应用窗口裁剪并更新元数据。

        始终保留：
        1. 第一条系统消息（如果存在）
        2. 最近 ``max_turns * 2`` 条非系统消息
        """
        # 窗口裁剪
        trimmed = self._trim_window(messages)
        trimmed_dicts = [message_to_dict(m) for m in trimmed]

        # 序列化写入 Redis
        self.store.set(session_id, SUB_KEY_MESSAGES, json.dumps(trimmed_dicts))

        # 更新元数据
        self._update_metadata(session_id, messages, trimmed)

        turn_count = sum(1 for m in messages if _msg_type(m) == "human")
        trimmed_turns = sum(1 for m in trimmed if _msg_type(m) == "human")
        logger.info(
            "MEM SAVE  session=%s  %d turns → stored %d (windowed)",
            session_id,
            turn_count,
            trimmed_turns,
        )

    def load_metadata(self, session_id: str) -> dict[str, Any]:
        """加载会话元数据，新会话返回空 dict。"""
        raw = self.store.get(session_id, SUB_KEY_METADATA)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("MEM LOAD  session=%s  metadata corrupted", session_id)
            return {}

    def clear(self, session_id: str) -> None:
        """清除会话的所有记忆（消息 + 元数据）。"""
        self.store.delete(session_id, SUB_KEY_MESSAGES)
        self.store.delete(session_id, SUB_KEY_METADATA)
        logger.info("MEM CLEAR session=%s", session_id)

    def get_message_count(self, session_id: str) -> int:
        """返回会话历史消息总数。"""
        meta = self.load_metadata(session_id)
        return meta.get("total_messages", 0)

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否已有存储的消息。"""
        return self.store.exists(session_id, SUB_KEY_MESSAGES)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_window(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """窗口裁剪：保留系统消息 + 最近 N 轮。"""
        if not messages:
            return messages

        # 判断第一条是否为系统消息
        keep_first = 1 if _msg_type(messages[0]) == "system" else 0
        tail = messages[keep_first:]
        max_tail = self.max_turns * 2  # 每轮 = human + ai 共 2 条

        if len(tail) <= max_tail:
            return messages

        trimmed = messages[:keep_first] + tail[-max_tail:]
        dropped = len(messages) - len(trimmed)
        logger.debug(
            "MEM TRIM  dropped %d oldest messages  (keep_first=%d, max_tail=%d)",
            dropped,
            keep_first,
            max_tail,
        )
        return trimmed

    def _update_metadata(
        self,
        session_id: str,
        all_messages: list[BaseMessage],
        stored: list[BaseMessage],
    ) -> None:
        """更新并持久化会话元数据。"""
        meta = self.load_metadata(session_id)
        now = datetime.now(tz=timezone.utc).isoformat()

        # 首次创建时写入初始字段（避坑 3：可信度字段）
        if "created_at" not in meta:
            meta.update({
                "source_type": "conversation",
                "confidence": 1.0,
                "created_at": now,
                "version": 1,
                "status": "active",
            })

        meta["updated_at"] = now
        meta["total_messages"] = len(all_messages)
        meta["total_turns"] = sum(1 for m in all_messages if _msg_type(m) == "human")
        meta["stored_messages"] = len(stored)

        self.store.set(session_id, SUB_KEY_METADATA, json.dumps(meta))


# ------------------------------------------------------------------
# Module helper
# ------------------------------------------------------------------

def _msg_type(msg: BaseMessage) -> str:
    """安全获取消息类型（兼容 dict 和 BaseMessage）。"""
    if hasattr(msg, "type"):
        return msg.type
    return msg.get("type", "") if isinstance(msg, dict) else ""
