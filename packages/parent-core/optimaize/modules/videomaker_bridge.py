from __future__ import annotations

# Copyright (c) 2026 OptimAIze.
# Bridge between the OptimAIze parent and the OptimAIze-VideoMaker child module.
#
# VideoMaker ships interchangeable backend implementations under
# ``modules/OptimAIze-VideoMaker/impl/<lang>`` (ts, go), all serving the same
# HTTP API on port 8003, plus a shared ``frontend``. The active backend is chosen
# with ``OPTIMAIZE_VIDEOMAKER_IMPL`` (default ``ts`` — the Remotion-native one).
# This bridge discovers the module, reports status, and launches the selected
# backend as an independent subprocess.

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def _discover_project_root() -> Path:
    configured = os.getenv("OPTIMAIZE_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "modules" / "OptimAIze-VideoMaker").is_dir():
            return parent
    return here.parents[4]


PROJECT_ROOT = _discover_project_root()
VM_CHILD_DIR = PROJECT_ROOT / "modules" / "OptimAIze-VideoMaker"
VM_IMPL_ROOT = VM_CHILD_DIR / "impl"
VM_FRONTEND_DIR = VM_CHILD_DIR / "frontend"
DEFAULT_IMPL = os.getenv("OPTIMAIZE_VIDEOMAKER_IMPL", "ts").strip().lower()


def _impl_dir(impl: str) -> Path:
    return VM_IMPL_ROOT / impl


@dataclass(frozen=True)
class VideoMakerChildStatus:
    exists: bool
    child_path: str
    impl: str
    impl_available: bool
    available_impls: list[str]
    frontend_available: bool
    message: str


def _available_impls() -> list[str]:
    if not VM_IMPL_ROOT.is_dir():
        return []
    return sorted(p.name for p in VM_IMPL_ROOT.iterdir() if p.is_dir())


def child_status(impl: str | None = None) -> VideoMakerChildStatus:
    impl = (impl or DEFAULT_IMPL).strip().lower()
    impls = _available_impls()
    impl_available = _impl_dir(impl).is_dir()
    frontend_available = (VM_FRONTEND_DIR / "index.html").exists()
    exists = VM_CHILD_DIR.exists()
    if exists and impl_available:
        message = f"OptimAIze-VideoMaker is available; impl '{impl}' can run independently."
    elif exists:
        message = (
            f"OptimAIze-VideoMaker exists, but impl '{impl}' was not found. "
            f"Available: {', '.join(impls) or 'none'}."
        )
    else:
        message = "OptimAIze-VideoMaker child project was not found."
    return VideoMakerChildStatus(
        exists=exists,
        child_path=str(VM_CHILD_DIR),
        impl=impl,
        impl_available=impl_available,
        available_impls=impls,
        frontend_available=frontend_available,
        message=message,
    )


def child_status_json() -> str:
    return json.dumps(asdict(child_status()), indent=2, ensure_ascii=False)


def _launch_command(impl: str, impl_dir: Path, server_port: int) -> tuple[list[str], Path] | None:
    """Return (command, cwd) to start the selected backend, or None if it can't.

    Each backend serves the same API on the chosen port via
    ``OPTIMAIZE_VIDEOMAKER_PORT``.
    """
    if impl == "ts":
        node = shutil.which("node")
        if not node:
            return None
        entry = impl_dir / "dist" / "server.js"
        if entry.exists():
            return ([node, str(entry)], impl_dir)
        return ([node, "--experimental-strip-types", str(impl_dir / "src" / "server.ts")], impl_dir)
    if impl == "go":
        binary = impl_dir / "optimaize-videomaker.exe"
        if binary.exists():
            return ([str(binary)], impl_dir)
        go = shutil.which("go")
        if go:
            return ([go, "run", "./cmd/optimaize-videomaker"], impl_dir)
        return None
    return None


def launch_child_api(server_port: int = 8003, impl: str | None = None) -> dict[str, object]:
    """Start the selected backend's API as an independent subprocess."""
    impl = (impl or DEFAULT_IMPL).strip().lower()
    status = child_status(impl)
    if not status.exists or not status.impl_available:
        return {"ok": False, "pid": None, "url": None, "message": status.message}

    launch = _launch_command(impl, _impl_dir(impl), server_port)
    if launch is None:
        return {
            "ok": False, "pid": None, "url": None,
            "message": f"No runnable command for impl '{impl}' (missing toolchain or build).",
        }
    cmd, cwd = launch
    env = dict(os.environ)
    env["OPTIMAIZE_VIDEOMAKER_PORT"] = str(server_port)
    process = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    return {
        "ok": True,
        "pid": process.pid,
        "url": f"http://127.0.0.1:{server_port}",
        "impl": impl,
        "message": f"Started OptimAIze-VideoMaker ({impl}) API on port {server_port}.",
    }
