"""Memory Gatekeeper —— 结构化记忆准入与生命周期管理。

在 ConversationMemory（原始消息窗口）之上提供结构化记忆条目的：
- 准入过滤（规则 + LLM 双阶段）
- 分类提取（6 种记忆类型）
- 冲突检测 & 去重合并
- 完整 CRUD + 生命周期管理

典型用法::

    gatekeeper = MemoryGatekeeper(entry_store, classifier)
    entries = gatekeeper.process_turn(
        session_id="demo-20240701",
        turn_index=3,
        user_message="我喜欢简短的回答",
        assistant_message="好的，我会尽量简洁。",
        tool_info=TurnToolInfo(),
    )
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.memory.classifier import MemoryClassifier
from app.core.memory.entry_models import (
    MemoryCandidate,
    MemoryEntry,
    MemoryEntryStatus,
    MemoryEntryType,
    TurnToolInfo,
)
from app.core.memory.session_store import RedisSessionStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 1: 规则预过滤（正则 / 启发式规则，不调用 LLM）
# ---------------------------------------------------------------------------

# 敏感信息正则（PII / 密钥）
_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API Key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("身份证号", re.compile(r"\d{17}[\dXx]")),
    ("手机号", re.compile(r"1[3-9]\d{9}")),
    ("JWT Token", re.compile(r"eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+")),
    ("密码赋值", re.compile(r"(?:password|passwd|pwd|token)\s*[=:]\s*\S+", re.IGNORECASE)),
    ("邮箱", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
]

# 纯命令前缀
_COMMAND_PREFIXES = ("/clear", "/memory", "/help", "/stats")

# 一次性临时查询关键词（仅含这些词且无后续约束 → 拦截）
_TEMP_QUERY_MARKERS = ("今天", "现在", "当前时间", "几点了", "几号")

# 极短消息无信息量阈值
_MIN_MEANINGFUL_LENGTH = 5


def _contains_sensitive_info(text: str) -> tuple[bool, str]:
    """检测文本是否包含敏感信息。

    Returns:
        (has_sensitive, reason) — 包含时为 (True, 匹配到的类型)。
    """
    for label, pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            return True, label
    return False, ""


def _is_command(text: str) -> bool:
    """检测是否为纯命令输入。"""
    stripped = text.strip().lower()
    return stripped.startswith(_COMMAND_PREFIXES)


def _is_trivial(text: str) -> bool:
    """检测是否为极短无信息量消息（纯招呼、语气词等）。"""
    stripped = text.strip()
    if len(stripped) < _MIN_MEANINGFUL_LENGTH:
        return True
    # 纯招呼词
    greeting_only = {"hi", "hello", "你好", "嗨", "在吗", "好", "嗯", "哦", "谢谢", "thanks", "ok"}
    if stripped.lower() in greeting_only:
        return True
    return False


def _is_likely_temporary(user_msg: str, assistant_msg: str) -> bool:
    """启发式判断是否为一次性临时查询（无长期记忆价值）。

    判断标准：用户消息是临时性疑问（天气、时间等），且助手回复不含
    可供未来参考的持久信息。
    """
    combined = (user_msg + " " + assistant_msg).strip()
    # 仅包含临时查询关键词
    has_temp_marker = any(m in user_msg for m in _TEMP_QUERY_MARKERS)
    # 不含长期偏好标记
    long_term_markers = ("以后", "一直", "总是", "习惯", "配置", "规范", "记住", "偏好", "纠正", "不对")
    has_long_term = any(m in combined for m in long_term_markers)
    return has_temp_marker and not has_long_term


def rule_based_filter(
    user_message: str,
    assistant_message: str,
) -> tuple[bool, str]:
    """Phase 1 规则预过滤 —— 快速拦截明显不应存储的对话。

    Args:
        user_message: 用户本轮输入。
        assistant_message: Agent 本轮最终回复。

    Returns:
        (should_skip, reason):
        - should_skip=True → 跳过后续 LLM 分类，不存储
        - reason 说明拦截原因
    """
    # 1. 纯命令
    if _is_command(user_message):
        return True, "纯命令输入 (/clear, /help 等)"

    # 2. 敏感信息
    has_sensitive, label = _contains_sensitive_info(user_message)
    if has_sensitive:
        return True, f"包含敏感信息: {label}"
    has_sensitive, label = _contains_sensitive_info(assistant_message)
    if has_sensitive:
        return True, f"助手回复中含敏感信息: {label}"

    # 3. 极短无信息量消息
    if _is_trivial(user_message):
        return True, "极短无信息量消息（<5 字或纯招呼）"

    # 4. 一次性临时查询（无长期标记）
    if _is_likely_temporary(user_message, assistant_message):
        return True, "一次性临时查询（天气/时间等，无长期价值）"

    return False, ""


# ---------------------------------------------------------------------------
# Phase 3: 冲突检测 & 去重
# ---------------------------------------------------------------------------

def _jaccard_similarity(a: str, b: str) -> float:
    """计算两个字符串的 Jaccard 相似度（基于字符 bigram）。"""
    if not a or not b:
        return 0.0

    def bigrams(s: str) -> set[str]:
        s = s.lower()
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

    set_a = bigrams(a)
    set_b = bigrams(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


_CONFLICT_THRESHOLD = 0.65
_DEDUP_THRESHOLD = 0.85
_CONFIDENCE_MARGIN = 0.2  # 新版置信度需比旧版高此值才能自动胜出


def _detect_similar(
    candidate: MemoryCandidate,
    existing_entries: list[MemoryEntry],
) -> tuple[MemoryEntry | None, str]:
    """检测候选是否与已有条目冲突或重复。

    Returns:
        (matched_entry, action):
        - (None, "new")          → 无冲突，新建条目
        - (entry, "duplicate")   → 几乎相同，应更新现有条目
        - (entry, "conflict")    → 语义相近但不完全相同，创建冲突组
    """
    best_entry: MemoryEntry | None = None
    best_score = 0.0

    # 1. 先在相同 entry_type 中找匹配
    candidate_type = candidate.memory_entry_type
    for entry in existing_entries:
        if entry.status not in (MemoryEntryStatus.active, MemoryEntryStatus.pending_review):
            continue
        if entry.entry_type != candidate_type:
            continue

        score_content = _jaccard_similarity(candidate.content, entry.content)
        score_summary = _jaccard_similarity(candidate.summary, entry.summary)
        score = max(score_content, score_summary)

        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None:
        return None, "new"

    if best_score >= _DEDUP_THRESHOLD:
        return best_entry, "duplicate"
    if best_score >= _CONFLICT_THRESHOLD:
        return best_entry, "conflict"

    return None, "new"


def _resolve_auto(
    candidate: MemoryCandidate,
    existing: MemoryEntry,
    action: str,
) -> tuple[MemoryEntry | None, MemoryEntry | None]:
    """自动解决冲突/去重。

    Returns:
        (to_save, to_archive):
        - to_save: 要持久化的条目（可能是新建、或更新后的 existing）
        - to_archive: 要归档的旧条目（冲突解决后）
    """
    if action == "duplicate":
        # 去重：更新 existing 的 updated_at，提升 confidence
        existing.confidence = max(existing.confidence, candidate.confidence)
        existing.bump_version()
        # 记录被再次提及
        existing.metadata["mention_count"] = existing.metadata.get("mention_count", 1) + 1
        logger.info("GATEKEEPER  dedup  entry=%s  confidence=%.2f", existing.entry_id, existing.confidence)
        return existing, None

    if action == "conflict":
        # 冲突：如新版置信度显著更高，新版胜出；否则标记 pending_review
        if candidate.confidence >= existing.confidence + _CONFIDENCE_MARGIN:
            # 新版胜出
            conflict_group = existing.conflict_group or f"cg-{uuid.uuid4().hex[:8]}"
            existing.archive()
            existing.conflict_group = conflict_group
            new_entry = MemoryEntry.from_candidate(candidate)
            new_entry.conflict_group = conflict_group
            logger.info(
                "GATEKEEPER  conflict-auto-resolved  winner=%s  loser=%s  group=%s",
                new_entry.entry_id,
                existing.entry_id,
                conflict_group,
            )
            return new_entry, existing
        else:
            # 无法自动裁决，标记为冲突
            conflict_group = f"cg-{uuid.uuid4().hex[:8]}"
            new_entry = MemoryEntry.from_candidate(candidate)
            new_entry.mark_status(MemoryEntryStatus.conflicted)
            new_entry.conflict_group = conflict_group
            existing.mark_status(MemoryEntryStatus.conflicted)
            existing.conflict_group = conflict_group
            logger.info(
                "GATEKEEPER  conflict-pending-review  entries=%s,%s  group=%s",
                new_entry.entry_id,
                existing.entry_id,
                conflict_group,
            )
            return new_entry, existing

    return None, None


# ---------------------------------------------------------------------------
# MemoryGatekeeper
# ---------------------------------------------------------------------------

class MemoryGatekeeper:
    """结构化记忆准入守卫。

    每个实例绑定一个 Redis 命名空间（``mem:entry``），负责：
    - 规则预过滤（拦截敏感/噪声）
    - LLM 分类提取（6 种记忆类型）
    - 冲突检测 & 去重合并
    - 条目 CRUD + 生命周期管理
    """

    # Redis sub-keys
    _SUB_DATA = "data"
    _SUB_IDX = "idx"
    _SUB_GIDX = "gidx"
    _SUB_CONFLICT = "conflict"

    def __init__(
        self,
        store: RedisSessionStore,
        classifier: MemoryClassifier | None = None,
    ) -> None:
        """初始化 Gatekeeper。

        Args:
            store: Redis 存储实例（应使用 key_prefix="mem:entry"）。
            classifier: LLM 分类器。为 None 时跳过 Phase 2，仅做规则过滤。
        """
        self.store = store
        self.classifier = classifier

    # ==================================================================
    # Public API — 主入口
    # ==================================================================

    def process_turn(
        self,
        session_id: str,
        turn_index: int,
        user_message: str,
        assistant_message: str,
        tool_info: TurnToolInfo | None = None,
    ) -> list[MemoryEntry]:
        """处理一轮对话，提取并持久化结构化记忆条目。

        这是 Gatekeeper 的主入口，完整执行双阶段管线。

        Args:
            session_id: 会话 ID。
            turn_index: 本轮在会话中的序号（从 0 开始）。
            user_message: 用户本轮输入文本。
            assistant_message: Agent 本轮最终回复文本。
            tool_info: 本轮工具调用摘要（可选）。

        Returns:
            本轮新持久化的 MemoryEntry 列表（不含去重更新的已有条目）。
        """
        # ---- Phase 1: 规则预过滤 ----
        should_skip, skip_reason = rule_based_filter(user_message, assistant_message)
        if should_skip:
            logger.debug("GATEKEEPER  phase1-skip  %s", skip_reason)
            return []

        # ---- Phase 2: LLM 分类提取 ----
        candidates: list[MemoryCandidate] = []
        if self.classifier is not None:
            tool_text = self._format_tool_info(tool_info)
            candidates = self.classifier.extract(user_message, assistant_message, tool_text)

        if not candidates:
            return []

        # ---- Phase 3: 冲突检测 & 持久化 ----
        existing = self.list_entries(session_id)
        saved: list[MemoryEntry] = []

        for candidate in candidates:
            if candidate.should_discard:
                continue

            # 3a. 冲突检测
            matched, action = _detect_similar(candidate, existing)

            if action == "new":
                # 新条目
                entry = MemoryEntry.from_candidate(
                    candidate,
                    source_session_id=session_id,
                    source_turn=turn_index,
                )
                self._persist_entry(entry, session_id)
                saved.append(entry)
                existing.append(entry)  # 加入现有列表，防止同批次重复
            else:
                # 去重或冲突
                to_save, to_archive = _resolve_auto(candidate, matched, action)
                if to_save:
                    # 对于 duplicate，to_save 是更新后的 existing（已在 _resolve_auto 中修改）
                    self._persist_entry(to_save, session_id)
                    if action == "duplicate":
                        # duplicate 不加入 saved（不是新条目）
                        pass
                    else:
                        saved.append(to_save)
                if to_archive:
                    self._persist_entry(to_archive, session_id)

        if saved:
            logger.info(
                "GATEKEEPER  session=%s  saved=%d new entries",
                session_id,
                len(saved),
            )
        return saved

    # ==================================================================
    # Public API — CRUD
    # ==================================================================

    def add_entry(self, entry: MemoryEntry, session_id: str = "") -> MemoryEntry:
        """直接写入一条记忆条目（跳过分类器和冲突检测）。"""
        self._persist_entry(entry, session_id or entry.source_session_id)
        logger.info("GATEKEEPER  added  entry=%s  type=%s", entry.entry_id, entry.entry_type.value)
        return entry

    def update_entry(self, entry_id: str, updates: dict[str, Any]) -> MemoryEntry | None:
        """更新条目字段，自动递增版本号。

        Returns:
            更新后的 MemoryEntry，条目不存在时返回 None。
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            return None

        # 仅允许更新部分字段
        allowed = {"content", "summary", "confidence", "expires_at", "tags", "metadata", "status"}
        for key, value in updates.items():
            if key in allowed and hasattr(entry, key):
                setattr(entry, key, value)

        entry.bump_version()
        self._persist_entry(entry, entry.source_session_id)
        logger.info("GATEKEEPER  updated  entry=%s  version=%d", entry_id, entry.version)
        return entry

    def delete_entry(self, entry_id: str) -> bool:
        """软删除条目（标记为 expired）。

        Returns:
            True 表示成功标记，False 表示条目不存在。
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            return False
        entry.mark_status(MemoryEntryStatus.expired)
        self._persist_entry(entry, entry.source_session_id)
        logger.info("GATEKEEPER  deleted  entry=%s", entry_id)
        return True

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """读取单条记忆条目。"""
        raw = self.store.get(entry_id, self._SUB_DATA)
        if not raw:
            return None
        try:
            return MemoryEntry(**json.loads(raw))
        except Exception as exc:
            logger.warning("GATEKEEPER  get_entry deserialize failed  %s: %s", entry_id, exc)
            return None

    def list_entries(
        self,
        session_id: str,
        *,
        status_filter: set[MemoryEntryStatus] | None = None,
    ) -> list[MemoryEntry]:
        """列出某会话的所有记忆条目。

        Args:
            session_id: 会话 ID。
            status_filter: 状态过滤器，默认仅返回 active + pending_review + conflicted。
        """
        if status_filter is None:
            status_filter = {
                MemoryEntryStatus.active,
                MemoryEntryStatus.pending_review,
                MemoryEntryStatus.conflicted,
            }

        raw = self.store.get(session_id, self._SUB_IDX)
        if not raw:
            return []

        try:
            entry_ids: list[str] = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("GATEKEEPER  list_entries index corrupted  session=%s", session_id)
            return []

        entries: list[MemoryEntry] = []
        for eid in entry_ids:
            entry = self.get_entry(eid)
            if entry and entry.status in status_filter:
                entries.append(entry)
        return entries

    def list_by_type(
        self,
        entry_type: MemoryEntryType,
        *,
        status_filter: set[MemoryEntryStatus] | None = None,
    ) -> list[MemoryEntry]:
        """跨会话按类型查询记忆条目（全局索引）。"""
        if status_filter is None:
            status_filter = {MemoryEntryStatus.active, MemoryEntryStatus.pending_review}

        raw = self.store.get(entry_type.value, self._SUB_GIDX)
        if not raw:
            return []

        try:
            entry_ids: list[str] = json.loads(raw)
        except json.JSONDecodeError:
            return []

        entries: list[MemoryEntry] = []
        for eid in entry_ids:
            entry = self.get_entry(eid)
            if entry and entry.status in status_filter:
                entries.append(entry)
        return entries

    def mark_for_review(self, entry_id: str) -> bool:
        """标记条目为待人工审核。"""
        entry = self.get_entry(entry_id)
        if entry is None:
            return False
        entry.mark_status(MemoryEntryStatus.pending_review)
        self._persist_entry(entry, entry.source_session_id)
        logger.info("GATEKEEPER  mark-review  entry=%s", entry_id)
        return True

    def resolve_conflict(
        self,
        conflict_group: str,
        winner_id: str,
        reviewer: str = "",
    ) -> list[MemoryEntry]:
        """解决冲突组：选出胜者，其余条目归档。

        Args:
            conflict_group: 冲突组 ID。
            winner_id: 胜出的条目 ID。
            reviewer: 审核人标识。

        Returns:
            该冲突组内所有条目（含胜者和已归档条目）。
        """
        raw = self.store.get(conflict_group, self._SUB_CONFLICT)
        if not raw:
            logger.warning("GATEKEEPER  conflict group not found: %s", conflict_group)
            return []

        try:
            entry_ids: list[str] = json.loads(raw)
        except json.JSONDecodeError:
            return []

        now = datetime.now(tz=timezone.utc).isoformat()
        resolved: list[MemoryEntry] = []

        for eid in entry_ids:
            entry = self.get_entry(eid)
            if entry is None:
                continue
            if eid == winner_id:
                entry.status = MemoryEntryStatus.active
                entry.conflict_group = None
                entry.reviewed_by = reviewer or "auto"
                entry.reviewed_at = now
            else:
                entry.mark_status(MemoryEntryStatus.archived)
                entry.reviewed_by = reviewer or "auto"
                entry.reviewed_at = now
            entry.bump_version()
            self._persist_entry(entry, entry.source_session_id)
            resolved.append(entry)

        logger.info(
            "GATEKEEPER  conflict-resolved  group=%s  winner=%s  total=%d",
            conflict_group,
            winner_id,
            len(resolved),
        )
        return resolved

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _persist_entry(self, entry: MemoryEntry, session_id: str) -> None:
        """将 MemoryEntry 写入 Redis 并更新所有索引。

        执行三步：
        1. 写 data key
        2. 更新 session index
        3. 更新 global type index
        4. 如果冲突，更新 conflict index
        """
        # 1. Data
        self.store.set(entry.entry_id, self._SUB_DATA, entry.model_dump_json(ensure_ascii=False))

        # 2. Session index
        self._update_index(session_id, entry.entry_id)

        # 3. Global type index
        self._update_index(entry.entry_type.value, entry.entry_id, is_global=True)

        # 4. Conflict index
        if entry.conflict_group:
            self._update_index(entry.conflict_group, entry.entry_id, is_conflict=True)

    def _update_index(
        self,
        index_key: str,
        entry_id: str,
        *,
        is_global: bool = False,
        is_conflict: bool = False,
    ) -> None:
        """更新索引 key（追加 entry_id，去重）。"""
        if is_global:
            sub = self._SUB_GIDX
        elif is_conflict:
            sub = self._SUB_CONFLICT
        else:
            sub = self._SUB_IDX

        raw = self.store.get(index_key, sub)
        entry_ids: list[str] = json.loads(raw) if raw else []
        if entry_id not in entry_ids:
            entry_ids.append(entry_id)
            self.store.set(index_key, sub, json.dumps(entry_ids, ensure_ascii=False))

    @staticmethod
    def _format_tool_info(tool_info: TurnToolInfo | None) -> str:
        """格式化工具调用信息为 LLM 输入文本。"""
        if tool_info is None:
            return "（本轮无工具调用）"
        parts = [f"工具名: {tool_info.tool_name}"]
        if tool_info.args:
            parts.append(f"参数: {json.dumps(tool_info.args, ensure_ascii=False)}")
        parts.append(f"成功: {'是' if tool_info.success else '否'}")
        if tool_info.error_message:
            parts.append(f"错误: {tool_info.error_message}")
        if tool_info.result_preview:
            parts.append(f"结果: {tool_info.result_preview}")
        return " | ".join(parts)
