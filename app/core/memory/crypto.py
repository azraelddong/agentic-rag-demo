"""记忆内容加密/解密工具。

使用 Fernet 对称加密（AES-128-CBC + HMAC-SHA256 认证），
加密后的内容以 ``ENC:`` 前缀包装，便于加载时自动识别。

典型用法::

    from cryptography.fernet import Fernet
    from app.core.memory.crypto import encrypt_content, decrypt_content, get_fernet

    fernet = get_fernet(settings)
    encrypted = encrypt_content("我的密码是abc123", fernet)
    # → "ENC:gAAAAABl..."
    plaintext = decrypt_content(encrypted, fernet)
    # → "我的密码是abc123"
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

from cryptography.fernet import Fernet

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)

# 加密后的内容以此为前缀，用于识别是否已加密
_ENCRYPTION_PREFIX = "ENC:"

# Fernet 密钥的有效长度（base64 编码后 44 字符，解码后 32 字节）
_FERNET_KEY_BYTES = 32


def get_fernet(settings: Settings | None = None) -> Fernet | None:
    """获取或自动生成 Fernet 加密实例。

    优先级：
    1. 从 Settings.memory_encryption_key 读取已配置的密钥。
    2. 未配置时自动生成临时密钥（进程生命周期内一致），
       打印警告提醒用户将密钥写入 .env 持久化。

    Args:
        settings: 应用配置。为 None 时仅检查环境变量。

    Returns:
        Fernet 实例，创建失败时返回 None。
    """
    global _cached_fernet, _key_source

    # 已缓存的实例直接返回
    if _cached_fernet is not None:
        return _cached_fernet

    key: str | None = None

    # 1. 尝试从 Settings 读取
    if settings is not None:
        key = settings.memory_encryption_key

    # 2. 回退到环境变量
    if not key:
        key = os.getenv("MEMORY_ENCRYPTION_KEY", "")

    if key:
        try:
            # 验证是否为合法的 Fernet key（urlsafe-base64 编码的 32 字节）
            decoded = base64.urlsafe_b64decode(key.encode("utf-8"))
            if len(decoded) != _FERNET_KEY_BYTES:
                logger.warning(
                    "CRYPTO  MEMORY_ENCRYPTION_KEY length mismatch: got %d bytes, need %d. "
                    "Generating new key.",
                    len(decoded),
                    _FERNET_KEY_BYTES,
                )
                key = None
        except Exception:
            logger.warning(
                "CRYPTO  MEMORY_ENCRYPTION_KEY is not a valid Fernet key. Generating new key."
            )
            key = None

    # 3. 自动生成
    if not key:
        key = Fernet.generate_key().decode("utf-8")
        _key_source = "auto-generated"
        logger.warning(
            "CRYPTO  No valid MEMORY_ENCRYPTION_KEY found. Auto-generated temporary key.\n"
            "        ⚠️  加密仅在当前进程生命周期内有效！重启后旧数据无法解密。\n"
            "        请将以下配置写入 .env 以持久化密钥：\n"
            "        MEMORY_ENCRYPTION_KEY=%s",
            key,
        )
    else:
        _key_source = "configured"

    try:
        _cached_fernet = Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except Exception as exc:
        logger.error("CRYPTO  Failed to initialize Fernet: %s", exc)
        return None

    logger.info("CRYPTO  Fernet initialized (key_source=%s)", _key_source)
    return _cached_fernet


# 进程级缓存
_cached_fernet: Fernet | None = None
_key_source: str = "none"


def encrypt_content(plaintext: str, fernet: Fernet) -> str:
    """加密文本内容。

    Args:
        plaintext: 明文内容。
        fernet: Fernet 加密实例。

    Returns:
        ``ENC:<base64_fernet_token>`` 格式的密文。
    """
    if not plaintext:
        return plaintext
    token = fernet.encrypt(plaintext.encode("utf-8"))
    return _ENCRYPTION_PREFIX + token.decode("utf-8")


def decrypt_content(wrapped: str, fernet: Fernet) -> str:
    """解密文本内容（自动识别是否加密）。

    Args:
        wrapped: 可能是 ``ENC:...`` 格式的密文，也可能是明文。
        fernet: Fernet 解密实例。

    Returns:
        解密后的明文。不满足加密格式时直接返回原文。
        解密失败时记录警告并返回原文（避免崩溃）。
    """
    if not wrapped or not wrapped.startswith(_ENCRYPTION_PREFIX):
        return wrapped

    token = wrapped[len(_ENCRYPTION_PREFIX):]
    try:
        decrypted = fernet.decrypt(token.encode("utf-8"))
        return decrypted.decode("utf-8")
    except Exception as exc:
        logger.warning(
            "CRYPTO  Decryption failed (token length=%d): %s. Returning raw ciphertext.",
            len(token),
            exc,
        )
        return wrapped


def is_encrypted(text: str) -> bool:
    """判断文本是否已经过加密包装。

    Args:
        text: 待检查的文本。

    Returns:
        True 如果以 ``ENC:`` 开头。
    """
    return text.startswith(_ENCRYPTION_PREFIX) if text else False


def needs_encryption(text: str) -> bool:
    """判断文本是否包含需要加密的敏感信息。

    复用 gatekeeper 中的 ``_contains_sensitive_info`` 正则检测逻辑。
    如果内容含 PII/密钥等敏感信息，返回 True。

    Args:
        text: 待检查的文本内容。

    Returns:
        True 如果文本包含疑似敏感信息。
    """
    if not text or not text.strip():
        return False
    # 延迟导入避免循环依赖
    from app.core.memory.gatekeeper import _contains_sensitive_info

    has_sensitive, _ = _contains_sensitive_info(text)
    return has_sensitive
