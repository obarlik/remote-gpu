"""Reusable, API-defined job blueprints for frequently-run job shapes.

A template is just data (task + defaults + required_params) — nothing
hardcoded in source, so any job shape the server already supports
(custom_script or a built-in task) can be turned into a template via the
API. Projects snapshot a template's values at creation time rather than
referencing it live, so editing a template never changes past projects.
"""
import time
from typing import Any

from gpu_server.config import DATA_DIR
from gpu_server.store import JsonStore

_store = JsonStore(DATA_DIR / "templates.json")


class TemplateManager:
    def __init__(self):
        self._templates: dict[str, dict[str, Any]] = _store.load()

    def create(self, name: str, task: str, defaults: dict[str, Any], required_params: list[str]) -> dict:
        if name in self._templates:
            raise ValueError(f"template '{name}' already exists")
        record = {
            "name": name,
            "task": task,
            "defaults": defaults,
            "required_params": required_params,
            "version": 1,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._templates[name] = record
        _store.save(self._templates)
        return record

    def update(self, name: str, defaults: dict[str, Any] | None, required_params: list[str] | None) -> dict:
        record = self.get(name)
        if record is None:
            raise ValueError(f"template '{name}' not found")
        if defaults is not None:
            record["defaults"] = {**record["defaults"], **defaults}
        if required_params is not None:
            record["required_params"] = required_params
        record["version"] += 1
        record["updated_at"] = time.time()
        _store.save(self._templates)
        return record

    def get(self, name: str) -> dict | None:
        return self._templates.get(name)

    def list(self) -> list[dict]:
        return list(self._templates.values())

    def delete(self, name: str) -> bool:
        if name not in self._templates:
            return False
        del self._templates[name]
        _store.save(self._templates)
        return True


template_manager = TemplateManager()
