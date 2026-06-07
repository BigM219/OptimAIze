from __future__ import annotations

# Copyright (c) 2026 OptimAIze.
# Bridge between the OptimAIze parent and the OptimAIze-Work child module.
#
# OptimAIze-Work ships several interchangeable implementations under
# ``modules/OptimAIze-Work/impl/<lang>`` (python, ts, go, rust), each serving the
# same HTTP API on port 8002. The parent is language-agnostic at runtime: it only
# needs to know which impl to launch. The active impl is chosen with the
# ``OPTIMAIZE_WORK_IMPL`` environment variable (default: ``go``, the recommended
# production runtime). This bridge discovers the module, reports status, and
# launches the selected impl as an independent subprocess.

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


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
WORK_IMPL_ROOT = WORK_CHILD_DIR / "impl"
DEFAULT_IMPL = os.getenv("OPTIMAIZE_WORK_IMPL", "go").strip().lower()


def _impl_dir(impl: str) -> Path:
    return WORK_IMPL_ROOT / impl


@dataclass(frozen=True)
class WorkChildStatus:
    exists: bool
    child_path: str
    impl: str
    impl_available: bool
    available_impls: list[str]
    web_available: bool
    message: str


def _available_impls() -> list[str]:
    if not WORK_IMPL_ROOT.is_dir():
        return []
    return sorted(p.name for p in WORK_IMPL_ROOT.iterdir() if p.is_dir())


def child_status(impl: str | None = None) -> WorkChildStatus:
    impl = (impl or DEFAULT_IMPL).strip().lower()
    impls = _available_impls()
    impl_dir = _impl_dir(impl)
    impl_available = impl_dir.is_dir()
    web_available = (impl_dir / "web" / "index.html").exists()
    exists = WORK_CHILD_DIR.exists()
    if exists and impl_available:
        message = f"OptimAIze-Work is available; impl '{impl}' can run independently."
    elif exists:
        message = (
            f"OptimAIze-Work exists, but impl '{impl}' was not found. "
            f"Available: {', '.join(impls) or 'none'}."
        )
    else:
        message = "OptimAIze-Work child project was not found."
    return WorkChildStatus(
        exists=exists,
        child_path=str(WORK_CHILD_DIR),
        impl=impl,
        impl_available=impl_available,
        available_impls=impls,
        web_available=web_available,
        message=message,
    )


def child_status_json() -> str:
    return json.dumps(asdict(child_status()), indent=2, ensure_ascii=False)


def _launch_command(impl: str, impl_dir: Path, server_port: int) -> tuple[list[str], Path] | None:
    """Return (command, cwd) to start the given impl's API, or None if it can't.

    Each impl is self-contained and serves the same API on the chosen port via
    the ``OPTIMAIZE_WORK_PORT`` environment variable.
    """
    if impl == "python":
        api_dir = impl_dir / "apps" / "api"
        return (
            [
                sys.executable, "-m", "uvicorn", "optimaize_work_api.main:app",
                "--app-dir", str(api_dir), "--host", "127.0.0.1", "--port", str(server_port),
            ],
            impl_dir,
        )
    if impl == "ts":
        node = shutil.which("node")
        entry = impl_dir / "dist" / "server.js"
        if node and entry.exists():
            return ([node, str(entry)], impl_dir)
        # Fall back to running TypeScript directly when a build is absent.
        if node:
            return ([node, "--experimental-strip-types", str(impl_dir / "src" / "server.ts")], impl_dir)
        return None
    if impl == "go":
        # Prefer a prebuilt binary; otherwise `go run`.
        binary = impl_dir / "optimaize-work.exe"
        if binary.exists():
            return ([str(binary)], impl_dir)
        go = shutil.which("go")
        if go:
            return ([go, "run", "./cmd/optimaize-work"], impl_dir)
        return None
    if impl == "rust":
        for candidate in (
            impl_dir / "target" / "release" / "optimaize-work.exe",
            impl_dir / "target" / "debug" / "optimaize-work.exe",
        ):
            if candidate.exists():
                return ([str(candidate)], candidate.parent)
        cargo = shutil.which("cargo")
        if cargo:
            return ([cargo, "run", "--release"], impl_dir)
        return None
    return None


def launch_child_api(server_port: int = 8002, impl: str | None = None) -> dict[str, object]:
    """Start the selected impl's API server as an independent subprocess."""
    impl = (impl or DEFAULT_IMPL).strip().lower()
    status = child_status(impl)
    if not status.exists or not status.impl_available:
        return {"ok": False, "pid": None, "url": None, "message": status.message}

    impl_dir = _impl_dir(impl)
    launch = _launch_command(impl, impl_dir, server_port)
    if launch is None:
        return {
            "ok": False, "pid": None, "url": None,
            "message": f"No runnable command for impl '{impl}' (missing toolchain or build).",
        }
    cmd, cwd = launch
    env = dict(os.environ)
    env["OPTIMAIZE_WORK_PORT"] = str(server_port)
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "ok": True,
        "pid": process.pid,
        "url": f"http://127.0.0.1:{server_port}",
        "impl": impl,
        "message": f"Started OptimAIze-Work ({impl}) API on port {server_port}.",
    }
