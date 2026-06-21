import logging
import re
from datetime import date, timedelta
from logging.config import dictConfig
from logging.handlers import BaseRotatingHandler
from pathlib import Path

"""支持基于时间和大小的日志文件滚动，保留指定天数的日志文件"""
class SizeAndTimeRotatingFileHandler(BaseRotatingHandler):
    """Rotate a log file at midnight or before it exceeds the size limit."""

    def __init__(
        self,
        filename: str | Path,   # 
        max_bytes: int,
        retention_days: int,
        encoding: str = "utf-8",
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")
        if retention_days <= 0:
            raise ValueError("retention_days must be greater than zero")

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(str(path), mode="a", encoding=encoding, delay=True)
        self.max_bytes = max_bytes
        self.retention_days = retention_days
        self.current_date = (
            date.fromtimestamp(path.stat().st_mtime) if path.exists() else date.today()
        )
        self.archive_pattern = re.compile(
            rf"^{re.escape(path.stem)}\.(\d{{4}}-\d{{2}}-\d{{2}})\.(\d+)"
            rf"{re.escape(path.suffix)}$"
        )
        self._delete_expired_archives()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self.shouldRollover(record):
                self.doRollover()
            super().emit(record)
        except Exception:
            self.handleError(record)

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if date.today() != self.current_date:
            return True

        path = Path(self.baseFilename)
        if not path.exists():
            return False

        message = f"{self.format(record)}\n"
        return path.stat().st_size + len(message.encode(self.encoding or "utf-8")) > self.max_bytes

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None

        source = Path(self.baseFilename)
        if source.exists() and source.stat().st_size:
            source.replace(self._next_archive_path(self.current_date))

        self.current_date = date.today()
        self._delete_expired_archives()

    def _next_archive_path(self, log_date: date) -> Path:
        source = Path(self.baseFilename)
        index = 1
        while True:
            archive = source.with_name(
                f"{source.stem}.{log_date:%Y-%m-%d}.{index:03d}{source.suffix}"
            )
            if not archive.exists():
                return archive
            index += 1

    def _delete_expired_archives(self) -> None:
        source = Path(self.baseFilename)
        cutoff = date.today() - timedelta(days=self.retention_days - 1)
        for candidate in source.parent.iterdir():
            match = self.archive_pattern.match(candidate.name)
            if not match:
                continue
            try:
                archive_date = date.fromisoformat(match.group(1))
            except ValueError:
                continue
            if archive_date < cutoff:
                candidate.unlink()


def configure_logging(
    level: str = "INFO",
    log_file: str | Path = "logs/agentic-rag-demo.log",
    max_bytes: int = 20 * 1024 * 1024,
    retention_days: int = 180,
) -> None:
    """Configure process-wide console and rolling-file logging."""

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                },
                "file": {
                    "()": SizeAndTimeRotatingFileHandler,
                    "filename": log_file,
                    "max_bytes": max_bytes,
                    "retention_days": retention_days,
                    "formatter": "default",
                },
            },
            "root": {
                "handlers": ["console"],
                "level": level.upper(),
            },
        }
    )
