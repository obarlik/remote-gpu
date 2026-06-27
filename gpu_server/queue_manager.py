import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from gpu_server.config import JOBS_DIR, TRAIN_PYTHON_EXE
from gpu_server.jobs.registry import CUSTOM_SCRIPT_TASK, is_known_task, resolve_task_module


class Job:
    def __init__(self, task: str, params: dict[str, Any]):
        self.id = uuid.uuid4().hex[:12]
        self.task = task
        self.params = params
        self.status = "queued"
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error: str | None = None
        self.output_dir = JOBS_DIR / self.id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "log.txt"
        self.process: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "status": self.status,
            "params": self.params,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "output_dir": str(self.output_dir),
        }


class JobQueue:
    """Sequential job queue: at most one training job runs at a time."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._pending: "queue.Queue[str]" = queue.Queue()
        self._order: list[str] = []
        self._global_lock = threading.Lock()
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()

    def submit(self, task: str, params: dict[str, Any]) -> Job:
        if not is_known_task(task):
            raise ValueError(f"Unknown task '{task}'")
        if task == CUSTOM_SCRIPT_TASK and not Path(params.get("script_path", "")).is_file():
            raise ValueError("custom_script task requires an existing 'script_path' param")
        job = Job(task, params)
        with self._global_lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._pending.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._global_lock:
            return [self._jobs[jid] for jid in self._order]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        with job.lock:
            if job.status == "queued":
                job.status = "cancelled"
                return True
            if job.status == "running" and job.process is not None:
                job.status = "cancelled"
                job.process.terminate()
                return True
        return False

    def _worker_loop(self) -> None:
        # Any uncaught exception here would silently kill the worker thread
        # and freeze the queue forever, so every job is isolated in a
        # try/except: one broken job must never take down the server.
        while True:
            job_id = self._pending.get()
            job = self._jobs[job_id]
            if job.status == "cancelled":
                continue
            try:
                self._run_job(job)
            except Exception as exc:
                with job.lock:
                    job.status = "failed"
                    job.error = f"server-side error launching job: {exc}"
                    job.finished_at = time.time()

    @staticmethod
    def _build_command(job: "Job", params_path: Path) -> list[str]:
        common_args = ["--params", str(params_path), "--output-dir", str(job.output_dir)]
        if job.task == CUSTOM_SCRIPT_TASK:
            script_path = job.params["script_path"]
            return [TRAIN_PYTHON_EXE, script_path, *common_args]
        module = resolve_task_module(job.task)
        return [TRAIN_PYTHON_EXE, "-m", module, *common_args]

    def _run_job(self, job: Job) -> None:
        params_path = job.output_dir / "params.json"
        params_path.write_text(json.dumps(job.params))
        cmd = self._build_command(job, params_path)

        with job.lock:
            job.status = "running"
            job.started_at = time.time()

        with open(job.log_path, "w", encoding="utf-8") as log_file:
            job.process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            return_code = job.process.wait()

        with job.lock:
            job.finished_at = time.time()
            if job.status == "cancelled":
                pass
            elif return_code == 0:
                job.status = "completed"
            else:
                job.status = "failed"
                job.error = f"process exited with code {return_code}"


job_queue = JobQueue()
