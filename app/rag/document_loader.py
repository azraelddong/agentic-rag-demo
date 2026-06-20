from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParsedDocument:
    """Normalized document content with source metadata."""

    text: str
    metadata: dict


class DocumentLoader:
    """Load supported documents from local files."""

    def __init__(self, allowed_extensions: list[str]) -> None:
        self.allowed_extensions = {ext.lower().lstrip(".") for ext in allowed_extensions}

    def load_directory(self, docs_dir: Path) -> list[ParsedDocument]:
        if not docs_dir.exists():
            logger.info("Docs directory does not exist, creating: %s", docs_dir)
            docs_dir.mkdir(parents=True, exist_ok=True)
            return []

        documents: list[ParsedDocument] = []
        for path in sorted(docs_dir.rglob("*")):
            if path.is_file() and path.suffix.lower().lstrip(".") in self.allowed_extensions:
                documents.append(self.load_file(path))
        return documents

    def load_file(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in self.allowed_extensions:
            raise DocumentProcessingError(f"不支持的文档类型: {path.suffix}")

        try:
            if suffix in {"txt", "md"}:
                text = path.read_text(encoding="utf-8")
            elif suffix == "pdf":
                text = self._read_pdf(path)
            else:
                raise DocumentProcessingError(f"文档类型暂未实现: {suffix}")
        except UnicodeDecodeError as exc:
            raise DocumentProcessingError(
                f"文档编码解析失败: {path.name}",
                detail={"file_path": str(path)},
            ) from exc
        except DocumentProcessingError:
            raise
        except Exception as exc:
            logger.exception("Failed to load document: %s", path)
            raise DocumentProcessingError(
                f"文档解析失败: {path.name}",
                detail={"file_path": str(path), "error": str(exc)},
            ) from exc

        cleaned = text.strip()
        if not cleaned:
            raise DocumentProcessingError(
                f"文档内容为空: {path.name}",
                detail={"file_path": str(path)},
            )

        created_at = datetime.now(UTC).isoformat()
        return ParsedDocument(
            text=cleaned,
            metadata={
                "file_name": path.name,
                "file_path": str(path.resolve()),
                "source_type": suffix,
                "created_at": created_at,
            },
        )

    def _read_pdf(self, path: Path) -> str:
        reader = PdfReader(str(path))
        page_texts: list[str] = []
        for page_index, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_texts.append(f"[page={page_index + 1}]\n{page_text}")
        return "\n\n".join(page_texts)
