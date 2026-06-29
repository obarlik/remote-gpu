import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
import pty
import select
import errno
from pathlib import Path
from typing import Any

from gpu_server.config import DATA_DIR, JOBS_DIR, TRAIN_PYTHON_EXE
from gpu_server.jobs.registry import CUSTOM_SCRIPT_TASK, is_known_task, resolve_task_module
from gpu_server.history_store import PartitionedHistoryStore

_history = PartitionedHistoryStore(DATA_DIR / "history")


class Job:
    def __init__(
        self,
        task: str,
        params: dict[str, Any],
        project: str | None = None,
        capabilities: list[str] | None = None,
        label: str | None = None,
        retention: dict[str, Any] | None = None,
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
        self.retention = retention or {
            "on_success": "keep_all",
            "on_failure": "keep_all",
            "on_cancelled": "keep_all",
            "ttl_hours": None
        }
        self.lock = threading.Lock()
        
        # Virtual Terminal & WebSocket properties
        self.ws_queues = set()
        self.ws_lock = threading.Lock()
        self.terminal_backlog = bytearray()
        self.master_fd = None

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
        job.retention = record.get("retention") or {
            "on_success": "keep_all",
            "on_failure": "keep_all",
            "on_cancelled": "keep_all",
            "ttl_hours": None
        }
        saved_path = Path(record["output_dir"])
        if not saved_path.exists():
            resolved = JOBS_DIR / record["id"]
            job.output_dir = resolved if resolved.exists() else saved_path
        else:
            job.output_dir = saved_path
        job.log_path = job.output_dir / "log.txt"
        job.process = None
        job.lock = threading.Lock()
        
        # Historical jobs terminal properties
        job.ws_queues = set()
        job.ws_lock = threading.Lock()
        job.terminal_backlog = bytearray()
        job.master_fd = None
        return job

    def to_dict(self) -> dict[str, Any]:
        ret_val = self.retention
        if ret_val is not None:
            if hasattr(ret_val, "dict"):
                ret_val = ret_val.dict()
            elif hasattr(ret_val, "model_dump"):
                ret_val = ret_val.model_dump()
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
            "retention": ret_val,
        }

    def send_input(self, data: str) -> None:
        """Writes interactive keyboard input data (keystrokes) to the PTY master."""
        with self.lock:
            if self.master_fd is not None:
                try:
                    os.write(self.master_fd, data.encode("utf-8"))
                except Exception as e:
                    print(f"Failed to write input to job {self.id}: {e}")

    def broadcast_output(self, data: bytes) -> None:
        """Appends output data to backlog and forwards it to all active WebSocket clients."""
        with self.ws_lock:
            self.terminal_backlog.extend(data)
            if len(self.terminal_backlog) > 100000:
                self.terminal_backlog = self.terminal_backlog[-100000:]
            
            for q in list(self.ws_queues):
                try:
                    q.put_nowait(data)
                except Exception:
                    pass


class JobQueue:
    """Sequential job queue: at most one training job runs at a time."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._pending: "queue.Queue[str]" = queue.Queue()
        self._order: list[str] = []
        self._global_lock = threading.Lock()
        self._history_loaded = False
        
        # Start history loading in a background thread to prevent startup block
        history_loader = threading.Thread(target=self._load_history_async, daemon=True)
        history_loader.start()
        
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()
        
        cleaner = threading.Thread(target=self._retention_cleaner_loop, daemon=True)
        cleaner.start()

    def _load_history_async(self) -> None:
        try:
            # Lossless automatic migration from unified jobs_history.json to partitioned history
            legacy_history_path = DATA_DIR / "jobs_history.json"
            if legacy_history_path.is_file():
                print("Migration: Found legacy jobs_history.json. Migrating to partitioned history...")
                try:
                    legacy_data = json.loads(legacy_history_path.read_text(encoding="utf-8"))
                    for key, record in legacy_data.items():
                        _history.save_one(key, record)
                    backup_path = legacy_history_path.with_suffix(".json.bak")
                    legacy_history_path.replace(backup_path)
                    print(f"Migration: Successfully migrated {len(legacy_data)} jobs. Legacy backup saved as jobs_history.json.bak")
                except Exception as migrate_err:
                    print(f"Migration failed: {migrate_err}")

            # Execute the heavy disk read outside of the global lock
            # so the main thread and queue submission are never blocked
            history_data = _history.load()
            records = sorted(history_data.values(), key=lambda r: r["created_at"])
            
            for record in records:
                job = Job.from_record(record)
                with self._global_lock:
                    self._jobs[job.id] = job
                    # Prevent duplicates in case jobs are added before history loading finishes
                    if job.id not in self._order:
                        self._order.append(job.id)
        except Exception as e:
            # Prevent crashes if history loading fails
            print(f"Error loading job history: {e}")
        finally:
            self._history_loaded = True

    def submit(
        self,
        task: str,
        params: dict[str, Any],
        project: str | None = None,
        capabilities: list[str] | None = None,
        label: str | None = None,
        retention: dict[str, Any] | None = None,
    ) -> Job:
        if not is_known_task(task):
            raise ValueError(f"Unknown task '{task}'")
        if task == CUSTOM_SCRIPT_TASK and not Path(params.get("script_path", "")).is_file():
            raise ValueError("custom_script task requires an existing 'script_path' param")
        job = Job(task, params, project, capabilities, label, retention)
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
                self._apply_immediate_cleanup(job)
                continue
            try:
                self._run_job(job)
            except Exception as exc:
                with job.lock:
                    job.status = "failed"
                    job.error = f"server-side error launching job: {exc}"
                    job.finished_at = time.time()
                _history.save_one(job.id, job.to_dict())
                self._apply_immediate_cleanup(job)

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

        repo_root = Path(__file__).resolve().parent.parent
        child_env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")])),
        }

        # Open pseudoterminal master/slave pair
        master_fd, slave_fd = pty.openpty()
        job.master_fd = master_fd

        popen_kwargs = {
            "stdin": slave_fd,
            "stdout": slave_fd,
            "stderr": slave_fd,
            "cwd": str(job.output_dir),
            "env": child_env,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True

        try:
            job.process = subprocess.Popen(cmd, **popen_kwargs)
        except Exception as exc:
            os.close(master_fd)
            os.close(slave_fd)
            job.master_fd = None
            raise exc

        # Close slave_fd in parent process so it receives EOF when child closes it
        os.close(slave_fd)

        # Background PTY output reader loop
        total_log_bytes = 0
        MAX_LOG_BYTES = 10 * 1024 * 1024  # 10MB limit to prevent disk bloating
        log_capped_msg_written = False

        with open(job.log_path, "wb") as log_file:
            while True:
                # Wait for PTY master_fd to become readable
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in r:
                    try:
                        data = os.read(master_fd, 8192)
                        if not data:
                            break  # EOF
                    except OSError as e:
                        if e.errno == errno.EIO:
                            break
                        raise e

                    # Broadcast output bytes to all active terminal websockets
                    job.broadcast_output(data)

                    # Write to disk log.txt up to 10MB limit
                    if total_log_bytes < MAX_LOG_BYTES:
                        log_file.write(data)
                        total_log_bytes += len(data)
                    elif not log_capped_msg_written:
                        cap_msg = b"\n*** LOG SIZE LIMIT EXCEEDED (10MB). FURTHER OUTPUT STREAMED ONLY VIA REAL-TIME WEBSOCKETS TO PREVENT DISK BLOAT ***\n"
                        log_file.write(cap_msg)
                        log_capped_msg_written = True

                # Check if process finished
                if job.process.poll() is not None:
                    # Drain any remaining bytes from master_fd
                    while True:
                        r_rem, _, _ = select.select([master_fd], [], [], 0.0)
                        if master_fd in r_rem:
                            try:
                                data = os.read(master_fd, 8192)
                                if not data:
                                    break
                                job.broadcast_output(data)
                                if total_log_bytes < MAX_LOG_BYTES:
                                    log_file.write(data)
                                    total_log_bytes += len(data)
                            except OSError as e:
                                if e.errno == errno.EIO:
                                    break
                                raise e
                        else:
                            break
                    break

        # Cleanup PTY master descriptor
        try:
            os.close(master_fd)
        except OSError:
            pass

        with job.lock:
            job.master_fd = None
            return_code = job.process.wait()
            job.finished_at = time.time()
            if job.status == "cancelled":
                pass
            elif return_code == 0:
                job.status = "completed"
            else:
                job.status = "failed"
                job.error = f"process exited with code {return_code}"
        _history.save_one(job.id, job.to_dict())
        self._apply_immediate_cleanup(job)

    def _apply_immediate_cleanup(self, job: Job) -> None:
        if not job.retention:
            return
        
        status = job.status  # "completed", "failed", "cancelled"
        ret = job.retention
        if hasattr(ret, "dict"):
            ret_dict = ret.dict()
        elif isinstance(ret, dict):
            ret_dict = ret
        else:
            ret_dict = {}

        policy = ret_dict.get(f"on_{status}") or ret_dict.get("on_completion")
        if not policy:
            if status == "completed":
                policy = ret_dict.get("on_success")
            elif status == "failed":
                policy = ret_dict.get("on_failure")
            elif status == "cancelled":
                policy = ret_dict.get("on_cancelled")

        if policy == "delete_all":
            try:
                import shutil
                if job.output_dir.exists():
                    shutil.rmtree(job.output_dir)
                    print(f"Immediate cleanup: Deleted output directory for job {job.id}")
            except Exception as e:
                print(f"Failed to delete all files for job {job.id}: {e}")
        elif policy == "delete_artifacts":
            try:
                keep_files = {"log.txt", "params.json", "metrics.jsonl"}
                if job.output_dir.exists():
                    for item in job.output_dir.iterdir():
                        if item.is_file() and item.name not in keep_files:
                            item.unlink()
                        elif item.is_dir():
                            import shutil
                            shutil.rmtree(item)
                    print(f"Immediate cleanup: Deleted non-log artifacts for job {job.id}")
            except Exception as e:
                print(f"Failed to delete artifacts for job {job.id}: {e}")

    def _retention_cleaner_loop(self) -> None:
        while True:
            # Scan every 5 minutes (300 seconds)
            time.sleep(300)
            now = time.time()
            jobs_to_clean = []
            
            with self._global_lock:
                for job in self._jobs.values():
                    if job.status not in ("completed", "failed", "cancelled"):
                        continue
                    if not job.retention:
                        continue
                    
                    ret = job.retention
                    if hasattr(ret, "dict"):
                        ret_dict = ret.dict()
                    elif isinstance(ret, dict):
                        ret_dict = ret
                    else:
                        ret_dict = {}
                        
                    ttl_hours = ret_dict.get("ttl_hours")
                    if ttl_hours is not None:
                        finished_at = job.finished_at or job.created_at
                        if now - finished_at > ttl_hours * 3600:
                            if job.output_dir.exists():
                                jobs_to_clean.append(job)
                                
            for job in jobs_to_clean:
                try:
                    import shutil
                    if job.output_dir.exists():
                        shutil.rmtree(job.output_dir)
                        print(f"TTL cleanup: Deleted output directory for job {job.id}")
                except Exception as e:
                    print(f"Failed to TTL clean job {job.id}: {e}")


job_queue = JobQueue()
