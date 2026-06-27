"""Resumable, chunkable upload sessions.

A client can send a file as one big chunk or split into many small ones —
the server just appends bytes at the offset the client says it's at, and
rejects writes that don't match (the client must resync via GET first).
This is independent of /v1/files, which stays for simple one-shot uploads.
"""
import gzip
import shutil
import threading
import uuid
from pathlib import Path

from gpu_server.config import DATASETS_DIR as UPLOADS_DIR

PENDING_DIR = UPLOADS_DIR / "_pending"
PENDING_DIR.mkdir(parents=True, exist_ok=True)


class UploadSession:
    def __init__(self, filename: str, gzip_encoded: bool):
        self.id = uuid.uuid4().hex[:12]
        self.filename = filename
        self.gzip_encoded = gzip_encoded
        self.tmp_path = PENDING_DIR / f"{self.id}.part"
        self.tmp_path.touch()
        self.received_bytes = 0
        self.lock = threading.Lock()

    def to_dict(self) -> dict:
        return {
            "upload_id": self.id,
            "filename": self.filename,
            "received_bytes": self.received_bytes,
        }


class UploadManager:
    def __init__(self):
        self._sessions: dict[str, UploadSession] = {}

    def start(self, filename: str, gzip_encoded: bool = False) -> UploadSession:
        safe_name = Path(filename or "upload.bin").name  # strip any path components
        session = UploadSession(safe_name, gzip_encoded)
        self._sessions[session.id] = session
        return session

    def get(self, upload_id: str) -> UploadSession | None:
        return self._sessions.get(upload_id)

    def append_chunk(self, session: UploadSession, offset: int, data: bytes) -> int:
        with session.lock:
            if offset != session.received_bytes:
                raise ValueError(f"offset mismatch: expected {session.received_bytes}, got {offset}")
            with open(session.tmp_path, "ab") as f:
                f.write(data)
            session.received_bytes += len(data)
            return session.received_bytes

    def complete(self, session: UploadSession) -> tuple[str, Path]:
        file_id = uuid.uuid4().hex[:12]
        file_dir = UPLOADS_DIR / file_id
        file_dir.mkdir(parents=True, exist_ok=True)

        dest = file_dir / session.filename
        if session.gzip_encoded:
            with gzip.open(session.tmp_path, "rb") as gz_in, open(dest, "wb") as out:
                shutil.copyfileobj(gz_in, out)
            session.tmp_path.unlink()
        else:
            session.tmp_path.replace(dest)

        del self._sessions[session.id]
        return file_id, dest

    def abort(self, session: UploadSession) -> None:
        session.tmp_path.unlink(missing_ok=True)
        del self._sessions[session.id]


upload_manager = UploadManager()
