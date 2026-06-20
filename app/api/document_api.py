from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_document_service
from app.schemas.document_schema import DocumentIndexRequest, DocumentIndexResponse, DocumentUploadResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    return await document_service.upload_document(file)


@router.post("/index", response_model=DocumentIndexResponse)
def index_documents(
    request: DocumentIndexRequest | None = None,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentIndexResponse:
    request = request or DocumentIndexRequest()
    return document_service.index_documents(
        docs_dir=request.docs_dir,
        rebuild=request.rebuild,
    )
