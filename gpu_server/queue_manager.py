import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from gpu_server.config import DATA_DIR, JOBS_DIR, TRAIN_PYTHON_EXE
from gpu_server.jobs.registry import CUSTOM_SCRIPT_TASK, is_known_task, resolve_task_module
from gpu_server.store import JsonStore

_history = JsonStore(DATA_DIR / "jobs_history.json")


class Job:
    def __init__(
        self,
        task: str,
        params: dict[str, Any],
        project: str | None = None,
        capabilities: list[str] | None = None,
        label: str | None = None,
    ):
        self.id = uuid.uuid4().hex[:12]
        self.task = task
        self.params = params
        self.project = project
        self.capabilities = capabilities or []
        self.label = label
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

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Job":
        """Rebuild a job's metadata from a persisted history record. Only
        used for historical jobs (always terminal), never re-run."""
        job = cls.__new__(cls)
        job.id = record["id"]
        job.task = record["task"]
        job.params = record["params"]
        job.project = record.get("project")
        job.capabilities = record.get("capabilities", [])
        job.label = record.get("label")
        job.status = record["status"]
        job.created_at = record["created_at"]
        job.started_at = record.get("started_at")
        job.finished_at = record.get("finished_at")
        job.error = record.get("error")
        saved_path = Path(record["output_dir"])
        if not saved_path.exists():
            resolved = JOBS_DIR / record["id"]
            job.output_dir = resolved if resolved.exists() else saved_path
        else:
            job.output_dir = saved_path
        job.log_path = job.output_dir / "log.txt"
        job.process = None
        job.lock = threading.Lock()
        return job

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "label": self.label,
            "project": self.project,
            "capabilities": self.capabilities,
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
        self._load_history()
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()

    def _load_history(self) -> None:
        # All history records are terminal jobs (only saved on completion),
        # so these are safe to restore as-is for listing.
        records = sorted(_history.load().values(), key=lambda r: r["created_at"])
        for record in records:
            job = Job.from_record(record)
            self._jobs[job.id] = job
            self._order.append(job.id)

    def submit(
        self,
        task: str,
        params: dict[str, Any],
        project: str | None = None,
        capabilities: list[str] | None = None,
        label: str | None = None,
    ) -> Job:
        if not is_known_task(task):
            raise ValueError(f"Unknown task '{task}'")
        if task == CUSTOM_SCRIPT_TASK and not Path(params.get("script_path", "")).is_file():
            raise ValueError("custom_script task requires an existing 'script_path' param")
        job = Job(task, params, project, capabilities, label)
        with self._global_lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._pending.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def set_project(self, job_id: str, project: str | None) -> Job | None:
        """Assign, reassign, or clear a job's project — works for jobs in
        any state, including already-finished ones (history is keyed by
        job id, so re-saving just overwrites that one record)."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        with job.lock:
            job.project = project
            is_terminal = job.status in ("completed", "failed", "cancelled")
        if is_terminal:
            _history.save_one(job.id, job.to_dict())
        return job

    def list_jobs_by_project(self, project: str) -> list[Job]:
        return [j for j in self.list_jobs() if j.project == project]

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
                self._terminate_process_tree(job.process)
                return True
        return False

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass

    def _worker_loop(self) -> None:
        # Any uncaught exception here would silently kill the worker thread
        # and freeze the queue forever, so every job is isolated in a
        # try/except: one broken job must never take down the server.
        while True:
            job_id = self._pending.get()
            job = self._jobs[job_id]
            if job.status == "cancelled":
                with job.lock:
                    job.finished_at = time.time()
                _history.save_one(job.id, job.to_dict())
                continue
            try:
                self._run_job(job)
            except Exception as exc:
                with job.lock:
                    job.status = "failed"
                    job.error = f"server-side error launching job: {exc}"
                    job.finished_at = time.time()
                _history.save_one(job.id, job.to_dict())

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
            if job.status == "cancelled":
                return
            job.status = "running"
            job.started_at = time.time()

        # Without this, a Python child on Windows whose stdout is redirected
        # to a file falls back to the system ANSI codepage (e.g. cp1252),
        # which can't encode non-ASCII text (Turkish, generated samples,
        # etc.) and crashes the job with UnicodeEncodeError instead of
        # logging it. Force UTF-8 regardless of locale.
        repo_root = Path(__file__).resolve().parent.parent
        child_env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            # Run with cwd=output_dir (below) so a script that writes a
            # relative path lands in its own job folder instead of polluting
            # the repo root — but built-in tasks still need `-m gpu_server.
            # jobs.X` to resolve, which relies on cwd normally; PYTHONPATH
            # keeps that working with cwd no longer pointing at the repo.
            "PYTHONPATH": os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")])),
        }

        with open(job.log_path, "w", encoding="utf-8") as log_file:
            popen_kwargs = {
                "stdout": log_file,
                "stderr": subprocess.STDOUT,
                "cwd": str(job.output_dir),
                "env": child_env,
            }
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True

            job.process = subprocess.Popen(cmd, **popen_kwargs)
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
        _history.save_one(job.id, job.to_dict())


job_queue = JobQueue()
