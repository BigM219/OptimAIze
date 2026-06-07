from __future__ import annotations

# Copyright (c) 2026 OptimAIze.
# Bridge between the OptimAIze parent and the OptimAIze-Work child module.
# Mirrors ocr_bridge.py: discover the workspace root robustly, report the
# child's status by checking its files exist, temporarily extend sys.path for
# in-process calls (lazy import), and launch the child's API as an independent
# subprocess.

import contextlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


def _discover_project_root() -> Path:
    """Locate the OptimAIze workspace root.

    Prefers ``OPTIMAIZE_PROJECT_ROOT``; then walks up looking for the module
    directory; finally falls back to a fixed depth.
    """
    configured = os.getenv("OPTIMAIZE_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "modules" / "OptimAIze-Work").is_dir():
            return parent
    return here.parents[4]


PROJECT_ROOT = _discover_project_root()
WORK_CHILD_DIR = PROJECT_ROOT / "modules" / "OptimAIze-Work"
WORK_CHILD_SRC = WORK_CHILD_DIR / "src"
WORK_API_DIR = WORK_CHILD_DIR / "apps" / "api"
WORK_WEB_DIR = WORK_CHILD_DIR / "apps" / "web"


@dataclass(frozen=True)
class WorkChildStatus:
    exists: bool
    child_path: str
    source_available: bool
    api_available: bool
    web_available: bool
    message: str


def child_status() -> WorkChildStatus:
    source_available = (WORK_CHILD_SRC / "optimaize_work" / "__init__.py").exists()
    api_available = (WORK_API_DIR / "optimaize_work_api" / "main.py").exists()
    web_available = (WORK_WEB_DIR / "index.html").exists()
    exists = WORK_CHILD_DIR.exists()
    if exists and source_available and api_available:
        message = "OptimAIze-Work is available and can run independently."
    elif exists:
        message = "OptimAIze-Work exists, but some expected files are missing."
    else:
        message = "OptimAIze-Work child project was not found."
    return WorkChildStatus(
        exists=exists,
        child_path=str(WORK_CHILD_DIR),
        source_available=source_available,
        api_available=api_available,
        web_available=web_available,
        message=message,
    )


def child_status_json() -> str:
    return json.dumps(asdict(child_status()), indent=2, ensure_ascii=False)


@contextlib.contextmanager
def child_import_path() -> Iterator[None]:
    paths = [str(WORK_CHILD_SRC), str(WORK_CHILD_DIR), str(WORK_API_DIR)]
    inserted = []
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
            inserted.append(path)
    try:
        yield
    finally:
        for path in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)


def create_sandbox(backend: str = "process", quota: dict | None = None) -> dict[str, object]:
    """Create a sandbox in-process via a lazy import of the child engine."""
    status = child_status()
    if not status.exists or not status.source_available:
        return {"ok": False, "message": status.message}

    with child_import_path():
        from optimaize_work.sandbox.manager import get_manager
        from optimaize_work.sandbox.types import SandboxQuota

        q = SandboxQuota(**quota) if quota else None
        info = get_manager().create(backend=backend, quota=q)
        return {"ok": True, "sandbox": info.to_dict(), "message": "Sandbox created."}


def launch_child_api(server_port: int = 8002) -> dict[str, object]:
    """Start the child's FastAPI server as an independent subprocess."""
    status = child_status()
    if not status.exists or not status.api_available:
        return {"ok": False, "pid": None, "url": None, "message": status.message}

    cmd = [
        sys.executable, "-m", "uvicorn",
        "optimaize_work_api.main:app",
        "--app-dir", str(WORK_API_DIR),
        "--host", "127.0.0.1",
        "--port", str(server_port),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=WORK_CHILD_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "ok": True,
        "pid": process.pid,
        "url": f"http://127.0.0.1:{server_port}",
        "message": f"Started OptimAIze-Work API on port {server_port}.",
    }
