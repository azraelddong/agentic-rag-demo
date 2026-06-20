from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.document_loader import ParsedDocument


@dataclass(slots=True)
class DocumentChunk:
    """Chunk content and metadata stored in the vector database."""

    content: str
    metadata: dict


class ChunkSplitter:
    """Split documents into chunks while preserving source metadata."""

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def split_documents(self, documents: list[ParsedDocument]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for document in documents:
            for index, text in enumerate(self.splitter.split_text(document.text)):
                if not text.strip():
                    continue
                metadata = {
                    **document.metadata,
                    "chunk_index": index,
                }
                chunks.append(DocumentChunk(content=text.strip(), metadata=metadata))
        return chunks
