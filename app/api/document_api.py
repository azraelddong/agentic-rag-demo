from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_document_service
from app.schemas.document_schema import DocumentIndexRequest, DocumentIndexResponse, DocumentUploadResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/api/documents", tags=["documents"])
"""文档管理接口，提供文档上传和索引功能。依赖注入DocumentService来处理业务逻辑。"""

"""上传文档接口，接收用户上传的文件并保存到服务器。"""
@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    return await document_service.upload_document(file)

"""索引文档接口，触发文档索引操作，将指定目录下的文档进行分块、嵌入并存储到向量数据库中。"""
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
