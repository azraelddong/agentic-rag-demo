from pathlib import Path

from app.core.config import Settings
from app.rag.document_loader import ParsedDocument
from app.rag.text_splitter import DocumentChunk
from app.services.document_service import DocumentService


class FakeLoader:
    def __init__(self, document: ParsedDocument) -> None:
        self.document = document

    def load_directory(self, docs_dir: Path) -> list[ParsedDocument]:
        return [self.document]


class FakeSplitter:
    def __init__(self, chunk: DocumentChunk) -> None:
        self.chunk = chunk

    def split_documents(self, documents: list[ParsedDocument]) -> list[DocumentChunk]:
        return [self.chunk]


class FakeEmbeddingModel:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.events.append("embed")
        return [[0.1, 0.2] for _ in texts]


class FakeVectorStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.deleted_paths: list[str] = []

    def drop_collection(self) -> None:
        self.events.append("drop")

    def delete_chunks_for_files(self, file_paths: list[str]) -> None:
        self.events.append("delete")
        self.deleted_paths = file_paths

    def add_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
        self.events.append("upsert")
        return len(chunks)


def test_index_documents_replaces_existing_file_chunks_before_upsert() -> None:
    file_path = r"E:\docs\demo.md"
    document = ParsedDocument(
        text="document content",
        metadata={
            "file_name": "demo.md",
            "file_path": file_path,
            "source_type": "md",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )
    chunk = DocumentChunk(content="document content", metadata={**document.metadata, "chunk_index": 0})
    events: list[str] = []
    vector_store = FakeVectorStore(events)
    service = DocumentService(
        settings=Settings(embedding_batch_size=64),
        loader=FakeLoader(document),
        splitter=FakeSplitter(chunk),
        embedding_model=FakeEmbeddingModel(events),
        vector_store=vector_store,
    )

    result = service.index_documents()

    assert vector_store.deleted_paths == [file_path]
    assert events == ["delete", "embed", "upsert"]
    assert result.inserted_count == 1
