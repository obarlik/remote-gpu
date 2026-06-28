"""Projects hold the durable state of a recurring experiment: which task,
which defaults (file paths + hyperparams), optionally snapshotted from a
template. Submitting a job against a project only needs to specify what's
different this run — everything else is inherited.
"""
from __future__ import annotations

import time
from typing import Any

from pathlib import Path

from gpu_server.config import DATA_DIR
from gpu_server.store import JsonStore
from gpu_server.templates import template_manager

_store = JsonStore(DATA_DIR / "projects.json")


class ProjectManager:
    def __init__(self):
        self._projects: dict[str, dict[str, Any]] = _store.load()

    def create(
        self,
        name: str,
        template: str | None,
        task: str | None,
        defaults: dict[str, Any],
        capabilities: list[str],
    ) -> dict:
        if name in self._projects:
            raise ValueError(f"project '{name}' already exists")

        resolved_task = task
        resolved_defaults: dict[str, Any] = {}
        required_params: list[str] = []
        resolved_capabilities = capabilities

        if template is not None:
            tpl = template_manager.get(template)
            if tpl is None:
                raise ValueError(f"template '{template}' not found")
            resolved_task = resolved_task or tpl["task"]
            resolved_defaults = dict(tpl["defaults"])
            required_params = list(tpl["required_params"])
            resolved_capabilities = list(tpl["capabilities"])

        if resolved_task is None:
            raise ValueError("task is required when no template is given")

        resolved_defaults.update(defaults)  # project-specific values win over template snapshot
        record = {
            "name": name,
            "template": template,
            "task": resolved_task,
            "defaults": resolved_defaults,
            "required_params": required_params,
            "capabilities": resolved_capabilities,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._projects[name] = record
        _store.save(self._projects)
        return record

    def update_defaults(self, name: str, defaults: dict[str, Any]) -> dict:
        record = self.get(name)
        if record is None:
            raise ValueError(f"project '{name}' not found")
        record["defaults"] = {**record["defaults"], **defaults}
        record["updated_at"] = time.time()
        _store.save(self._projects)
        return record

    def get(self, name: str) -> dict | None:
        return self._projects.get(name)

    def list(self) -> list[dict]:
        return list(self._projects.values())

    def delete(self, name: str) -> bool:
        if name not in self._projects:
            return False
        del self._projects[name]
        _store.save(self._projects)
        return True

    def list_files(self, name: str) -> list[dict[str, Any]]:
        """Surfaces which of a project's defaults are actual files on disk —
        these are what every job under the project inherits and shares,
        as opposed to plain hyperparam values."""
        record = self.get(name)
        if record is None:
            raise ValueError(f"project '{name}' not found")
        files = []
        for param, value in record["defaults"].items():
            if isinstance(value, str) and Path(value).is_file():
                files.append({"param": param, "path": value, "size_bytes": Path(value).stat().st_size})
        return files

    def resolve_job(self, name: str, overrides: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
        """Merge project defaults with this run's overrides and validate
        required_params. Returns (task, final_params, capabilities)."""
        record = self.get(name)
        if record is None:
            raise ValueError(f"project '{name}' not found")
        final_params = {**record["defaults"], **overrides}
        missing = [k for k in record["required_params"] if k not in final_params]
        if missing:
            raise ValueError(f"missing required params for project '{name}': {missing}")
        return record["task"], final_params, record["capabilities"]


project_manager = ProjectManager()
