from __future__ import annotations

from optimaize.modules import videomaker_bridge


def videomaker_status() -> dict[str, object]:
    status = videomaker_bridge.child_status()
    return {
        "id": "videomaker",
        "name": "OptimAIze-VideoMaker",
        "kind": "child-module",
        "available": status.exists and status.impl_available,
        "impl": status.impl,
        "impl_available": status.impl_available,
        "available_impls": status.available_impls,
        "frontend_available": status.frontend_available,
        "path": status.child_path,
        "message": status.message,
        "web_url": "http://127.0.0.1:8003",
        "api_url": "http://127.0.0.1:8003",
    }


def launch_api(server_port: int = 8003) -> dict[str, object]:
    return videomaker_bridge.launch_child_api(server_port=server_port)
