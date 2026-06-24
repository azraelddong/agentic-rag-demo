import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat_api import router as chat_router
from app.api.document_api import router as document_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging_config import configure_logging

"""获取settings配置信息"""
settings = get_settings()

"""配置日志记录，使用settings中的参数设置日志级别、文件路径、文件大小和保留天数。"""
configure_logging(
    level=settings.log_level,
    log_file=settings.log_path,
    max_bytes=settings.log_max_file_size_bytes,
    retention_days=settings.log_retention_days,
)
logger = logging.getLogger(__name__)

"""主应用模块，创建FastAPI实例，配置中间件和路由，并定义全局异常处理器。"""
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Enterprise-style Basic RAG demo with Agentic RAG extension points.",
)
"""添加CORS中间件，允许所有来源、方法和头部，以支持跨域请求。"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""路由注册，将文档相关的API路由和聊天相关的API路由包含到主应用中。"""
app.include_router(document_router)
app.include_router(chat_router)

"""健康检查接口，返回应用的状态、名称和环境信息。"""
@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
    }

"""全局异常处理器，捕获AppError类型的异常并返回结构化的JSON响应，同时记录警告日志。对于未预料的异常，记录错误日志并返回通用的服务器错误响应。"""
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("%s: %s", exc.code, exc.message, extra={"detail": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "detail": exc.detail,
        },
    )

"""全局异常处理器，捕获未预料的异常并返回通用的服务器错误响应，同时记录错误日志。"""
@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected server error")
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "服务器内部错误",
            "detail": {},
        },
    )
