import logging
from datetime import date, timedelta
from pathlib import Path

from app.core.logging_config import SizeAndTimeRotatingFileHandler


def test_rotates_before_a_log_file_exceeds_its_size_limit(tmp_path: Path) -> None:
    log_file = tmp_path / "app.log"
    handler = SizeAndTimeRotatingFileHandler(
        log_file, max_bytes=80, retention_days=180
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("test.size_rotation")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.info("a" * 60)
    logger.info("b" * 60)
    handler.close()

    archives = list(tmp_path.glob("app.*.001.log"))
    assert len(archives) == 1
    assert archives[0].read_text(encoding="utf-8") == f"{'a' * 60}\n"
    assert log_file.read_text(encoding="utf-8") == f"{'b' * 60}\n"


def test_removes_archives_older_than_retention_period(tmp_path: Path) -> None:
    old_date = date.today() - timedelta(days=2)
    old_archive = tmp_path / f"app.{old_date:%Y-%m-%d}.001.log"
    old_archive.write_text("expired\n", encoding="utf-8")

    handler = SizeAndTimeRotatingFileHandler(
        tmp_path / "app.log", max_bytes=80, retention_days=2
    )
    handler.close()

    assert not old_archive.exists()
