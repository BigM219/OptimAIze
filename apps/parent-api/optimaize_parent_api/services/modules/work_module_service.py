from __future__ import annotations

from optimaize.modules import work_bridge


def work_status() -> dict[str, object]:
    status = work_bridge.child_status()
    return {
        "id": "work",
        "name": "OptimAIze-Work",
        "kind": "child-module",
        "available": status.exists and status.source_available,
        "source_available": status.source_available,
        "api_available": status.api_available,
        "web_available": status.web_available,
        "path": status.child_path,
        "message": status.message,
        "web_url": "http://127.0.0.1:5180",
        "api_url": "http://127.0.0.1:8002",
    }


def launch_api(server_port: int = 8002) -> dict[str, object]:
    return work_bridge.launch_child_api(server_port=server_port)
