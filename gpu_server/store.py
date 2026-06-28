"""Tiny atomic JSON-file persistence — no database dependency, just durable
dicts on disk. Used for templates, projects, and job history so a server
restart doesn't lose them."""
import json
import threading
from pathlib import Path
from typing import Any


class JsonStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        with self.lock:
            if not self.path.exists():
                return {}
            return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, Any]) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def save_one(self, key: str, record: dict[str, Any]) -> None:
        """Load-modify-save a single key — used for state that can be
        revised after the fact (e.g. a job's project reassignment), where
        a plain append-only log would leave stale duplicate records."""
        data = self.load()
        data[key] = record
        self.save(data)
