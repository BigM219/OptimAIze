from __future__ import annotations


class OptimAIzeParentError(Exception):
    status_code = 400


class ModuleLaunchError(OptimAIzeParentError):
    status_code = 500
