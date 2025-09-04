from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.config import get_config
from src.api.v1.endpoints import router

config = get_config()

app = FastAPI(
    title=config.settings.app_name,
    version=config.settings.app_version,
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json"
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["System"])
async def root():
    return {
        "message": f"Welcome to {config.settings.app_name}",
        "version": config.settings.app_version,
        "classify": "/api/v1/classify",
        "status": "operational"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_type": "internal_server_error",
            "error_message": "An unexpected error occurred",
            "error_code": "UNHANDLED_EXCEPTION"
        }
    )


if __name__ == "__main__":
    api_config = config.get_api_config()

    uvicorn.run(
        "src.main:app",
        host=api_config["host"],
        port=api_config["port"],
        reload=api_config["reload"],
        log_level="info" if not config.settings.debug else "debug",
        access_log=True
    )