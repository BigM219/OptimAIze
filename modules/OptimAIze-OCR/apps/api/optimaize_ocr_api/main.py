from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from optimaize_ocr_api.api.v1 import routes_health, routes_history, routes_ocr
from optimaize_ocr_api.core.config import get_settings
from optimaize_ocr_api.core.errors import OptimAIzeOCRError

settings = get_settings()

app = FastAPI(title=settings.api_title, version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(OptimAIzeOCRError)
async def handle_domain_error(_: Request, exc: OptimAIzeOCRError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/health")
def root_health() -> dict[str, str | bool]:
    return {"ok": True, "service": settings.api_title, "version": settings.api_version}


app.include_router(routes_health.router, prefix="/api/v1")
app.include_router(routes_ocr.router, prefix="/api/v1")
app.include_router(routes_history.router, prefix="/api/v1")
