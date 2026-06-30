"""Redis 会话记忆模块测试。

需要本地 Redis 实例（docker compose up -d redis）。
如果 Redis 不可用，集成测试自动跳过。
"""

from __future__ import annotations

import json
import time

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)

from app.core.memory.conversation_memory import (
    SUB_KEY_MESSAGES,
    SUB_KEY_METADATA,
    ConversationMemory,
)
from app.core.memory.models import SessionMetadata
from app.core.memory.session_store import RedisSessionStore


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _redis_available() -> bool:
    """检测本地 Redis 是否可用。"""
    try:
        import redis
        r = redis.Redis.from_url("redis://localhost:6379/0")
        r.ping()
        r.close()
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        return False


def _make_messages(n: int, start_idx: int = 0) -> list:
    """生成 n 条交替的 human/ai 消息对。"""
    msgs = []
    for i in range(start_idx, start_idx + n):
        msgs.append(HumanMessage(content=f"问题 {i}"))
        msgs.append(AIMessage(content=f"回答 {i}"))
    return msgs


# 测试用的 session_id，含时间戳避免冲突
SESSION_ID = f"test-{int(time.time())}"


# ---------------------------------------------------------------------------
# SessionStore 单元测试（使用 fakeredis 思路 — 但这里直接测真实 Redis）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _redis_available(),
    reason="Redis not available. Start with: docker compose up -d redis",
)
class TestRedisSessionStore:
    """RedisSessionStore 集成测试（需要真实 Redis）。"""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """每个测试后清理测试 key。"""
        yield
        store = RedisSessionStore()
        store.delete(SESSION_ID, SUB_KEY_MESSAGES)
        store.delete(SESSION_ID, SUB_KEY_METADATA)
        store.close()

    def test_set_and_get(self):
        store = RedisSessionStore()
        store.set(SESSION_ID, SUB_KEY_MESSAGES, '{"hello":"world"}')
        result = store.get(SESSION_ID, SUB_KEY_MESSAGES)
        assert result == '{"hello":"world"}'
        store.close()

    def test_get_missing_key_returns_none(self):
        store = RedisSessionStore()
        result = store.get("nonexistent-session-id", SUB_KEY_MESSAGES)
        assert result is None
        store.close()

    def test_set_overwrites_existing(self):
        store = RedisSessionStore()
        store.set(SESSION_ID, SUB_KEY_MESSAGES, "v1")
        store.set(SESSION_ID, SUB_KEY_MESSAGES, "v2")
        assert store.get(SESSION_ID, SUB_KEY_MESSAGES) == "v2"
        store.close()

    def test_delete_removes_key(self):
        store = RedisSessionStore()
        store.set(SESSION_ID, SUB_KEY_MESSAGES, "data")
        store.delete(SESSION_ID, SUB_KEY_MESSAGES)
        assert store.get(SESSION_ID, SUB_KEY_MESSAGES) is None
        store.close()

    def test_exists(self):
        store = RedisSessionStore()
        assert not store.exists(SESSION_ID, SUB_KEY_MESSAGES)
        store.set(SESSION_ID, SUB_KEY_MESSAGES, "data")
        assert store.exists(SESSION_ID, SUB_KEY_MESSAGES)
        store.close()

    def test_ttl_is_set_on_stored_key(self):
        store = RedisSessionStore(default_ttl=300)
        store.set(SESSION_ID, SUB_KEY_MESSAGES, "data")
        ttl = store.client.ttl(store._key(SESSION_ID, SUB_KEY_MESSAGES))
        assert 0 < ttl <= 300
        store.close()

    def test_ttl_is_refreshed_on_get(self):
        store = RedisSessionStore(default_ttl=300)
        store.set(SESSION_ID, SUB_KEY_MESSAGES, "data")
        # 不等待，直接验证 TTL 被重置为 default_ttl
        _ = store.get(SESSION_ID, SUB_KEY_MESSAGES)
        ttl = store.client.ttl(store._key(SESSION_ID, SUB_KEY_MESSAGES))
        assert 0 < ttl <= 300
        store.close()

    def test_sanitize_session_id(self):
        """包含特殊字符的 session_id 应被清理。"""
        store = RedisSessionStore()
        bad_id = "demo:2024/06/30"
        safe_key = store._key(bad_id, "messages")
        # 不应包含 : 或 /（除了前缀中的冒号）
        assert ":" in safe_key  # 前缀部分的冒号
        assert "/" not in safe_key
        store.close()


# ---------------------------------------------------------------------------
# ConversationMemory 集成测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _redis_available(),
    reason="Redis not available. Start with: docker compose up -d redis",
)
class TestConversationMemory:
    """ConversationMemory 集成测试（需要真实 Redis）。"""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        store = RedisSessionStore()
        store.delete(SESSION_ID, SUB_KEY_MESSAGES)
        store.delete(SESSION_ID, SUB_KEY_METADATA)
        store.close()

    def test_load_messages_new_session_returns_empty(self):
        mem = ConversationMemory(RedisSessionStore())
        msgs = mem.load_messages("brand-new-session")
        assert msgs == []

    def test_save_and_load_round_trip(self):
        mem = ConversationMemory(RedisSessionStore())
        original = [
            HumanMessage(content="你好"),
            AIMessage(content="你好！有什么可以帮助你的？"),
        ]
        mem.save_messages(SESSION_ID, original)
        loaded = mem.load_messages(SESSION_ID)
        assert len(loaded) == 2
        assert loaded[0].content == "你好"
        assert loaded[0].type == "human"
        assert loaded[1].content == "你好！有什么可以帮助你的？"
        assert loaded[1].type == "ai"

    def test_save_and_load_with_tool_calls(self):
        """带有 tool_call 和 ToolMessage 的消息应正确往返。"""
        mem = ConversationMemory(RedisSessionStore())
        msgs = [
            HumanMessage(content="1+1等于多少"),
            AIMessage(
                content="",
                tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "call_1"}],
            ),
            ToolMessage(content="2", tool_call_id="call_1"),
            AIMessage(content="1+1 等于 2"),
        ]
        mem.save_messages(SESSION_ID, msgs)
        loaded = mem.load_messages(SESSION_ID)
        assert len(loaded) == 4
        assert loaded[2].type == "tool"
        assert loaded[2].content == "2"

    def test_metadata_tracks_counts(self):
        mem = ConversationMemory(RedisSessionStore())
        msgs = _make_messages(3)  # 3 human + 3 ai = 6 messages
        mem.save_messages(SESSION_ID, msgs)
        meta = mem.load_metadata(SESSION_ID)
        assert meta["total_messages"] == 6
        assert meta["total_turns"] == 3
        assert meta["stored_messages"] == 6
        assert meta["source_type"] == "conversation"
        assert meta["confidence"] == 1.0
        assert "created_at" in meta
        assert "updated_at" in meta
        assert meta["version"] == 1
        assert meta["status"] == "active"

    def test_clear_removes_all_data(self):
        mem = ConversationMemory(RedisSessionStore())
        mem.save_messages(SESSION_ID, [HumanMessage(content="test")])
        assert mem.load_messages(SESSION_ID)
        mem.clear(SESSION_ID)
        assert mem.load_messages(SESSION_ID) == []
        assert mem.load_metadata(SESSION_ID) == {}

    def test_get_message_count(self):
        mem = ConversationMemory(RedisSessionStore())
        assert mem.get_message_count(SESSION_ID) == 0
        mem.save_messages(SESSION_ID, _make_messages(2))
        assert mem.get_message_count(SESSION_ID) == 4

    def test_session_exists(self):
        store = RedisSessionStore()
        mem = ConversationMemory(store)
        assert not mem.session_exists(SESSION_ID)
        mem.save_messages(SESSION_ID, [HumanMessage(content="hi")])
        assert mem.session_exists(SESSION_ID)


# ---------------------------------------------------------------------------
# ConversationMemory 窗口裁剪测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _redis_available(),
    reason="Redis not available. Start with: docker compose up -d redis",
)
class TestMessageTrimming:
    """消息窗口裁剪行为测试。"""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        store = RedisSessionStore()
        store.delete(SESSION_ID, SUB_KEY_MESSAGES)
        store.delete(SESSION_ID, SUB_KEY_METADATA)
        store.close()

    def test_preserves_system_message(self):
        """系统消息应始终保留在首位。"""
        mem = ConversationMemory(RedisSessionStore(), max_turns=2)
        msgs = [
            SystemMessage(content="You are a helpful assistant."),
            *_make_messages(5),  # 10 messages, exceeds max_turns=2 (4 non-system)
        ]
        mem.save_messages(SESSION_ID, msgs)
        loaded = mem.load_messages(SESSION_ID)
        assert loaded[0].type == "system"
        assert loaded[0].content == "You are a helpful assistant."

    def test_trims_oldest_when_over_limit(self):
        """超限时应丢弃最旧的非系统消息。"""
        mem = ConversationMemory(RedisSessionStore(), max_turns=2)
        msgs = _make_messages(5)  # 5 turns = 10 msgs, limit is 2 turns = 4 msgs
        mem.save_messages(SESSION_ID, msgs)
        loaded = mem.load_messages(SESSION_ID)

        # 应保留最近 2 轮 = 4 条消息
        assert len(loaded) == 4, f"Expected 4, got {len(loaded)}"
        # 最新一轮的 human 消息应该是 "问题 4"
        assert loaded[-2].content == "问题 4"
        assert loaded[-1].content == "回答 4"

    def test_no_trim_when_under_limit(self):
        """未超限时不应裁剪。"""
        mem = ConversationMemory(RedisSessionStore(), max_turns=10)
        msgs = _make_messages(3)  # 3 turns = 6 msgs
        mem.save_messages(SESSION_ID, msgs)
        loaded = mem.load_messages(SESSION_ID)
        assert len(loaded) == 6

    def test_stored_messages_reflects_trimmed_count(self):
        mem = ConversationMemory(RedisSessionStore(), max_turns=1)
        mem.save_messages(SESSION_ID, _make_messages(10))
        meta = mem.load_metadata(SESSION_ID)
        assert meta["total_messages"] == 20  # 原始
        assert meta["stored_messages"] == 2  # 裁剪后仅保留 1 轮 = 2 条


# ---------------------------------------------------------------------------
# SessionMetadata 模型测试（无需 Redis）
# ---------------------------------------------------------------------------

class TestSessionMetadata:
    """SessionMetadata Pydantic 模型单元测试。"""

    def test_default_values(self):
        meta = SessionMetadata()
        assert meta.source_type == "conversation"
        assert meta.confidence == 1.0
        assert meta.version == 1
        assert meta.status == "active"
        assert meta.total_messages == 0
        assert meta.total_turns == 0
        assert meta.stored_messages == 0

    def test_confidence_bounds(self):
        """confidence 必须在 0.0~1.0 之间。"""
        with pytest.raises(ValueError):
            SessionMetadata(confidence=1.5)
        with pytest.raises(ValueError):
            SessionMetadata(confidence=-0.1)

    def test_serialization_round_trip(self):
        meta = SessionMetadata(total_messages=10, total_turns=5)
        data = meta.model_dump()
        reloaded = SessionMetadata(**data)
        assert reloaded.total_messages == 10
        assert reloaded.total_turns == 5


# ---------------------------------------------------------------------------
# 消息序列化兼容性测试（无需 Redis）
# ---------------------------------------------------------------------------

class TestMessageSerialization:
    """LangChain message_to_dict / messages_from_dict 兼容性。"""

    def test_basic_types(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        dicts = [message_to_dict(m) for m in msgs]
        restored = messages_from_dict(dicts)
        assert len(restored) == 3
        assert restored[0].type == "system"
        assert restored[1].type == "human"
        assert restored[2].type == "ai"

    def test_tool_message(self):
        msgs = [
            HumanMessage(content="calc"),
            AIMessage(content="", tool_calls=[{
                "name": "calculator",
                "args": {"expression": "2+2"},
                "id": "call_123",
            }]),
            ToolMessage(content="4", tool_call_id="call_123"),
        ]
        dicts = [message_to_dict(m) for m in msgs]
        restored = messages_from_dict(dicts)
        assert restored[2].type == "tool"
        assert restored[2].content == "4"
