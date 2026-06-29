"""Partitioned JSON-file history persistence.
Saves each job's metadata as an independent JSON file under data/history/<job_id>.json.
Prevents file bloating and reduces disk I/O compared to a single giant history file.
"""
import json
import threading
from pathlib import Path
from typing import Any


class PartitionedHistoryStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def save_one(self, key: str, record: dict[str, Any]) -> None:
        """Saves a single job record atomically to directory/<key>.json."""
        with self.lock:
            path = self.directory / f"{key}.json"
            tmp = path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
                tmp.replace(path)
            except Exception as e:
                print(f"Failed to save history for {key}: {e}")

    def load(self) -> dict[str, Any]:
        """Loads all job records from directory/*.json and returns a dict keyed by job ID."""
        data = {}
        with self.lock:
            for file_path in self.directory.glob("*.json"):
                try:
                    record = json.loads(file_path.read_text(encoding="utf-8"))
                    job_id = file_path.stem
                    data[job_id] = record
                except Exception as e:
                    print(f"Failed to load history file {file_path}: {e}")
        return data

    def delete(self, key: str) -> None:
        """Deletes the history file for the given key/job ID."""
        with self.lock:
            path = self.directory / f"{key}.json"
            try:
                if path.exists():
                    path.unlink()
            except Exception as e:
                print(f"Failed to delete history file for {key}: {e}")
