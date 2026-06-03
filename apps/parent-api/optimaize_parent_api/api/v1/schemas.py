from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str


class ModuleStatusResponse(BaseModel):
    id: str
    name: str
    kind: str
    available: bool
    source_available: bool
    ui_available: bool
    api_available: bool
    web_available: bool
    path: str
    message: str
    web_url: str | None = None
    api_url: str | None = None


class ModulesResponse(BaseModel):
    modules: list[ModuleStatusResponse]


class LaunchOCRUIRequest(BaseModel):
    server_port: int = 7860
    share: bool = False
    inbrowser: bool = False


class LaunchResultResponse(BaseModel):
    ok: bool
    pid: int | None = None
    url: str | None = None
    message: str
