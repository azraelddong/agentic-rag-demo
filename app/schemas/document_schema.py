from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    file_name: str
    file_path: str
    size_bytes: int
    status: str = "uploaded"


class DocumentIndexRequest(BaseModel):
    docs_dir: str | None = Field(
        default=None,
        description="Optional docs directory. Defaults to DOCS_DIR in .env.",
    )
    rebuild: bool = Field(
        default=False,
        description="Drop and recreate the Milvus collection before indexing.",
    )


class DocumentIndexResponse(BaseModel):
    indexed_files: int
    chunk_count: int
    inserted_count: int
    collection_name: str
    message: str
