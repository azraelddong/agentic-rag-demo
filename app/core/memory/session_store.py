"""Redis 底层会话存储封装。

提供带 TTL 自动续期的 key-value 存取，所有 key 以统一前缀命名空间隔离。
"""

from __future__ import annotations

import logging
import re

import redis

logger = logging.getLogger(__name__)

_VALID_SESSION_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def _sanitize_session_id(session_id: str) -> str:
    """移除可能破坏 Redis key 结构的特殊字符。"""
    return _VALID_SESSION_ID_RE.sub("", session_id)


class RedisSessionStore:
    """Redis 底层会话存储。

    每个值以 ``{prefix}:{session_id}:{sub_key}`` 形式存储，
    读写时自动刷新 TTL。

    Example::

        store = RedisSessionStore("redis://localhost:6379/0", default_ttl=3600)
        store.set("demo-20240630", "messages", '[{"type":"human",...}]')
        raw = store.get("demo-20240630", "messages")  # → str | None
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        key_prefix: str = "mem:session",
    ) -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _key(self, session_id: str, sub_key: str) -> str:
        safe = _sanitize_session_id(session_id)
        return f"{self.key_prefix}:{safe}:{sub_key}"

    def get(self, session_id: str, sub_key: str) -> str | None:
        """读取一个值，命中时自动刷新 TTL。返回 None 表示 key 不存在。"""
        full_key = self._key(session_id, sub_key)
        try:
            value = self.client.get(full_key)
        except redis.RedisError as exc:
            logger.error("MEM GET  %s  FAILED: %s", full_key, exc)
            raise
        if value is not None:
            self.client.expire(full_key, self.default_ttl)
            logger.debug("MEM GET  %s  (OK, %d chars)", full_key, len(value))
        else:
            logger.debug("MEM GET  %s  (MISS)", full_key)
        return value

    def set(self, session_id: str, sub_key: str, value: str) -> None:
        """写入一个值并设置 TTL。"""
        full_key = self._key(session_id, sub_key)
        try:
            self.client.set(full_key, value, ex=self.default_ttl)
        except redis.RedisError as exc:
            logger.error("MEM SET  %s  FAILED: %s", full_key, exc)
            raise
        logger.debug(
            "MEM SET  %s  (%d chars, TTL=%ds)",
            full_key,
            len(value),
            self.default_ttl,
        )

    def delete(self, session_id: str, sub_key: str) -> None:
        """删除一个 key。"""
        full_key = self._key(session_id, sub_key)
        try:
            self.client.delete(full_key)
        except redis.RedisError as exc:
            logger.error("MEM DEL  %s  FAILED: %s", full_key, exc)
            raise
        logger.debug("MEM DEL  %s", full_key)

    def exists(self, session_id: str, sub_key: str) -> bool:
        """检查 key 是否存在。"""
        full_key = self._key(session_id, sub_key)
        try:
            return bool(self.client.exists(full_key))
        except redis.RedisError as exc:
            logger.error("MEM EXISTS  %s  FAILED: %s", full_key, exc)
            raise

    def close(self) -> None:
        """关闭 Redis 连接。"""
        self.client.close()
        logger.debug("MEM  Redis connection closed")
