from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARENT_CORE = PROJECT_ROOT / "packages" / "parent-core"
if str(PARENT_CORE) not in sys.path:
    sys.path.insert(0, str(PARENT_CORE))

from optimaize_parent_api.api.v1 import routes_health, routes_modules
from optimaize_parent_api.core.config import get_settings
from optimaize_parent_api.core.errors import OptimAIzeParentError

settings = get_settings()

app = FastAPI(title=settings.api_title, version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(OptimAIzeParentError)
async def handle_domain_error(_: Request, exc: OptimAIzeParentError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/health")
def root_health() -> dict[str, str | bool]:
    return {"ok": True, "service": settings.api_title, "version": settings.api_version}


app.include_router(routes_health.router, prefix="/api/v1")
app.include_router(routes_modules.router, prefix="/api/v1")
