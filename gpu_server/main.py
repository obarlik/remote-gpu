import gzip
import json
import mimetypes
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from gpu_server.auth import require_token
from gpu_server.config import DATASETS_DIR as UPLOADS_DIR
from gpu_server.projects import project_manager
from gpu_server.queue_manager import job_queue
from gpu_server.routes_lab import router as lab_router
from gpu_server.schemas import (
    JobInfo,
    JobMoveRequest,
    JobSubmitRequest,
    UploadInitRequest,
    JobLogsResponse,
    GPUStatusResponse,
    JobFileInfoExtended,
)
from gpu_server.server_info import FEATURES, VERSION
from gpu_server.uploads import upload_manager

app = FastAPI(
    title="remote-gpu training server",
    description=(
        "Exposes this machine's GPU to remote clients for training jobs. "
        "All endpoints require `Authorization: Bearer <token>`. "
        "Submit a job with task='custom_script' to run any framework-agnostic "
        "training script (torch, OpenCL, raw CUDA kernels, ...), or task="
        "'transformer_train' for the built-in parametrized transformer LM.\n\n"
        "**For slow/unreliable links:** uploads can be sent in one shot "
        "(`POST /v1/files`) or resumed/chunked (`POST /v1/uploads` + `PUT "
        ".../{id}?offset=N` + `.../complete` — survives a dropped connection, "
        "see `GET /v1/uploads/{id}` to find the resume offset). Both upload "
        "paths support an explicit `gzip_encoded` flag to send compressed "
        "bytes. Downloads (`GET /v1/jobs/{id}/files/{filename}`) support "
        "`Range` requests for the same reason. Responses are gzip-compressed "
        "when the client sends `Accept-Encoding: gzip`.\n\n"
        "**For repeated/recurring job shapes:** define a `Template` (`POST "
        "/v1/templates` — task + defaults + required_params, pure data, not "
        "hardcoded) and create a `Project` from it (`POST /v1/projects` — "
        "snapshots the template's values, so later template edits don't "
        "affect existing projects). `POST /v1/projects/{name}/jobs` then "
        "only needs this run's deltas — everything else is inherited from "
        "the project. `PATCH /v1/projects/{name}` updates just the given "
        "keys (e.g. point at a newly uploaded dataset without re-stating "
        "everything). `GET /v1/projects/{name}/files` lists which defaults "
        "are actual files on disk (vs. plain hyperparams) — what every job "
        "under the project currently shares. `PATCH /v1/jobs/{id}` assigns/"
        "moves/unassigns any job's project after the fact, even a running "
        "one. File scope is per-job by default and never global — there's "
        "no 'latest artifact' registry, so reusing one job's output in "
        "another is always an explicit path, typically via a project's "
        "`defaults`. Job history, templates, and projects all persist "
        "across server restarts.\n\n"
        "**Capabilities (declarative, not guessed):** a job/template/project "
        "can declare `capabilities=['metrics']`, meaning its script writes "
        "`output_dir/metrics.jsonl` (any keys, e.g. `{\"step\":10,\"loss\":0.3}`) "
        "— `GET /v1/jobs/{id}/metrics` returns it, and the dashboard charts "
        "every key as its own line instead of guessing from raw log text. "
        "Declaring `'resume'` means the script reads a `resume_from` key from "
        "its params (a checkpoint path) and continues training from it if "
        "present — fixed convention, same idea as `metrics.jsonl`. To resume "
        "a run, submit a new job/project-run with `{\"resume_from\": "
        "\"<checkpoint path>\", ...}` in params.\n\n"
        "**Labeling and discoverability:** `task` is a fixed dispatch "
        "directive, not a description — set `\"label\"` on `POST /v1/jobs` "
        "or `POST /v1/projects/{name}/jobs` to give a run a free-form, "
        "human-readable name (e.g. `\"scaleup d640 attempt 3\"`), shown in "
        "job listings and the dashboard instead of everyone just seeing "
        "`custom_script`. `GET /v1/server-info` returns the server's "
        "version and a list of feature flags, so a client can check what's "
        "supported in one request instead of re-parsing this description "
        "after every update."
    ),
    version=VERSION,
)
# Compresses outgoing responses (job lists, logs, file downloads) when the
# client sends Accept-Encoding: gzip. Pure response-side, opt-in via that
# header, so clients that don't ask for it see no change at all.
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.include_router(lab_router)


def read_log_backwards(log_path: Path, limit: int, cursor: int | None = None) -> dict:
    if not log_path.exists():
        return {"lines": [], "next_cursor": None, "total_bytes": 0, "has_more": False}

    file_size = log_path.stat().st_size
    if file_size == 0:
        return {"lines": [], "next_cursor": None, "total_bytes": 0, "has_more": False}

    # Start from the cursor if valid, otherwise start from the end of the file
    start_pos = cursor if (cursor is not None and 0 <= cursor <= file_size) else file_size
    if start_pos == 0:
        return {"lines": [], "next_cursor": None, "total_bytes": file_size, "has_more": False}

    lines = []
    chunk_size = 4096
    buffer = b""
    pos = start_pos
    has_more = False

    with open(log_path, "rb") as f:
        while pos > 0 and len(lines) < limit:
            # Determine how much to read
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            buffer = chunk + buffer

            # Split buffer into lines
            line_parts = buffer.split(b"\n")
            
            # If pos > 0, the first part might be incomplete, keep it in the buffer
            if pos > 0:
                buffer = line_parts[0]
                chunk_lines = line_parts[1:]
            else:
                buffer = b""
                chunk_lines = line_parts

            # Add lines in reverse order (since we are moving backwards)
            for cl in reversed(chunk_lines):
                # Decode line, replace bad chars
                decoded = cl.decode("utf-8", errors="replace")
                lines.append(decoded)
                if len(lines) >= limit:
                    break

        # If we broke out because of limit and there is still content
        if pos > 0 or len(buffer) > 0:
            has_more = True
            # The next cursor is the byte offset we reached
            # If we had a partial line left in buffer, pos needs to account for it
            next_cursor = pos + len(buffer)
        else:
            next_cursor = None

    # Since we collected lines backwards, reverse them back to chronological order
    lines.reverse()

    return {
        "lines": lines,
        "next_cursor": next_cursor,
        "total_bytes": file_size,
        "has_more": has_more,
    }


_DASHBOARD_HTML = (Path(__file__).resolve().parent / "static" / "dashboard.html").read_text(encoding="utf-8")


@app.get("/v1/server-info", summary="Server version and supported feature flags")
def server_info():
    """Lets a client check what this server supports in one request instead
    of guessing from a version number or re-reading docs after every
    update. New entries only ever get appended to 'features' — never
    renamed or removed — so a client can safely check for one by name."""
    return {"version": VERSION, "features": FEATURES}


@app.get("/dashboard", response_class=HTMLResponse, summary="Web UI: job list, status, and a best-effort loss chart")
def dashboard():
    """Static page; it calls the same /v1/* API from the browser using a
    bearer token you enter once (stored in the browser's localStorage).
    Doesn't expose anything not already available via the API."""
    return _DASHBOARD_HTML


@app.get("/v1/gpu", response_model=GPUStatusResponse, summary="Current GPU name, VRAM, utilization, temperature, fan speed, and power")
def gpu_status(_: None = Depends(require_token)):
    """Queries nvidia-smi for a live snapshot of the GPU this server controls."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,fan.speed,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        lines = out.strip().splitlines()
        if not lines:
            raise ValueError("No GPU details returned by nvidia-smi")
        parts = [x.strip() for x in lines[0].split(",")]
        
        # Parse standard parameters
        name = parts[0]
        mem_total = int(parts[1])
        mem_used = int(parts[2])
        util = int(parts[3])
        
        # Helper to parse optional/non-numeric strings (like "[Not Supported]")
        def safe_int(val):
            try:
                return int(val)
            except ValueError:
                return None

        def safe_float(val):
            try:
                return float(val)
            except ValueError:
                return None

        temp = safe_int(parts[4]) if len(parts) > 4 else None
        fan = safe_int(parts[5]) if len(parts) > 5 else None
        power_draw = safe_float(parts[6]) if len(parts) > 6 else None
        power_limit = safe_float(parts[7]) if len(parts) > 7 else None

        return {
            "name": name,
            "memory_total_mb": mem_total,
            "memory_used_mb": mem_used,
            "utilization_pct": util,
            "temperature_c": temp,
            "fan_speed_pct": fan,
            "power_draw_w": power_draw,
            "power_limit_w": power_limit,
        }
    except FileNotFoundError:
        return {
            "name": "Mock/CPU Environment (nvidia-smi not found)",
            "memory_total_mb": 24564,
            "memory_used_mb": 0,
            "utilization_pct": 0,
            "temperature_c": 42,
            "fan_speed_pct": 30,
            "power_draw_w": 45.2,
            "power_limit_w": 250.0,
        }
    except Exception as exc:
        raise HTTPException(500, f"nvidia-smi query failed: {exc}") from exc



@app.post("/v1/files", summary="Upload a dataset, driver script, or kernel source")
def upload_file(file: UploadFile, gzip_encoded: bool = False, _: None = Depends(require_token)):
    """Generic upload endpoint: a dataset, a driver script, or a raw kernel
    source (.cl / .cu / .py) — anything a training job's script_path or
    params may need to reference by path. Returns {file_id, path}; pass
    `path` as script_path (or any custom param) when submitting a job.

    Set ?gzip_encoded=true if the request body itself is gzip-compressed
    (a transport-only encoding) — it's decompressed on arrival and stored
    under the filename you gave, unchanged. This is independent of the
    filename: a file that is itself a real .gz dataset and should be
    stored as-is must NOT set this flag, since the filename alone doesn't
    tell the server your intent."""
    file_id = uuid.uuid4().hex[:12]
    file_dir = UPLOADS_DIR / file_id
    file_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.bin").name  # strip any path components
    dest = file_dir / safe_name

    if gzip_encoded:
        with gzip.GzipFile(fileobj=file.file, mode="rb") as gz_in, open(dest, "wb") as out:
            shutil.copyfileobj(gz_in, out)
    else:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)

    return {"file_id": file_id, "path": str(dest)}


@app.post("/v1/uploads", summary="Start a resumable/chunked upload session")
def start_upload(req: UploadInitRequest, _: None = Depends(require_token)):
    """Use this instead of /v1/files when the link is slow or unreliable, or
    when the client wants to send the file in pieces rather than one shot.
    Returns {upload_id, filename, received_bytes: 0}. Then PUT chunks to
    /v1/uploads/{upload_id} in order, and POST .../complete when done."""
    session = upload_manager.start(req.filename, req.gzip_encoded)
    return session.to_dict()


@app.get("/v1/uploads/{upload_id}", summary="Check how many bytes an upload session has received")
def get_upload_status(upload_id: str, _: None = Depends(require_token)):
    """After a dropped connection, call this to find out where to resume:
    send the next chunk starting at the returned received_bytes offset."""
    session = upload_manager.get(upload_id)
    if session is None:
        raise HTTPException(404, "Upload session not found")
    return session.to_dict()


@app.put("/v1/uploads/{upload_id}", summary="Append one chunk at a given byte offset")
async def put_upload_chunk(upload_id: str, offset: int, request: Request, _: None = Depends(require_token)):
    """The chunk can be any size — a whole file in one PUT, or split into
    many small ones; offset must equal the session's current received_bytes
    (i.e. chunks are appended strictly in order, no gaps). If it doesn't
    match, the server returns 409 with the correct offset to retry at."""
    session = upload_manager.get(upload_id)
    if session is None:
        raise HTTPException(404, "Upload session not found")
    data = await request.body()
    try:
        received_bytes = upload_manager.append_chunk(session, offset, data)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"received_bytes": received_bytes}


@app.post("/v1/uploads/{upload_id}/complete", summary="Finalize an upload session into a usable file")
def complete_upload(upload_id: str, _: None = Depends(require_token)):
    """Returns {file_id, path} in the same shape as POST /v1/files, so the
    result can be used as script_path/params the same way either way."""
    session = upload_manager.get(upload_id)
    if session is None:
        raise HTTPException(404, "Upload session not found")
    file_id, dest = upload_manager.complete(session)
    return {"file_id": file_id, "path": str(dest)}


@app.delete("/v1/uploads/{upload_id}", summary="Abort an upload session and discard partial data")
def abort_upload(upload_id: str, _: None = Depends(require_token)):
    session = upload_manager.get(upload_id)
    if session is None:
        raise HTTPException(404, "Upload session not found")
    upload_manager.abort(session)
    return {"aborted": True}


@app.post("/v1/jobs", response_model=JobInfo, summary="Submit a training job to the queue")
def submit_job(req: JobSubmitRequest, _: None = Depends(require_token)):
    """Queues a job; it runs once any earlier jobs finish (one job at a time).
    For task='custom_script', params must include 'script_path' from a prior
    /v1/files upload. Returns the job with status='queued'."""
    try:
        job = job_queue.submit(req.task, req.params, capabilities=req.capabilities, label=req.label)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return job.to_dict()


@app.get("/v1/jobs", response_model=list[JobInfo], summary="List all jobs, oldest first")
def list_jobs(_: None = Depends(require_token)):
    return [job.to_dict() for job in job_queue.list_jobs()]


@app.get("/v1/jobs/{job_id}", response_model=JobInfo, summary="Get one job's status")
def get_job(job_id: str, _: None = Depends(require_token)):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.patch("/v1/jobs/{job_id}", response_model=JobInfo, summary="Assign, move, or unassign a job's project")
def move_job(job_id: str, req: JobMoveRequest, _: None = Depends(require_token)):
    """Works for a job in any state, including running or already finished
    — the project field is just a label for organizing/listing, the
    running process (if any) never sees or depends on it."""
    if req.project is not None and project_manager.get(req.project) is None:
        raise HTTPException(400, f"project '{req.project}' not found")
    job = job_queue.set_project(job_id, req.project)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.get(
    "/v1/jobs/{job_id}/logs",
    summary="Job stdout/stderr logs (supports plain text download or paginated JSON chunks)",
)
def get_job_logs(
    job_id: str,
    limit: int | None = Query(
        None,
        description="Limit the response to this many lines. If set, returns a JSON object with next_cursor and lines.",
    ),
    cursor: int | None = Query(
        None,
        description="Byte offset indicating where to begin reading backwards. Use the next_cursor from previous JSON response.",
    ),
    _: None = Depends(require_token),
):
    """
    Returns the stdout/stderr logs of the job.
    
    - If `limit` is not specified, returns the entire log file as `text/plain`.
    - If `limit` is specified, returns a `JobLogsResponse` JSON object with paginated chunks
      reading backwards from the end of the file or the specified `cursor` offset.
    """
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not job.log_path.exists():
        if limit is not None:
            return {"lines": [], "next_cursor": None, "total_bytes": 0, "has_more": False}
        return PlainTextResponse("")

    if limit is not None:
        # JSON Paginated Mode
        data = read_log_backwards(job.log_path, limit, cursor)
        return data
    else:
        # Plain text Backward-Compatible Mode (returns entire file)
        return PlainTextResponse(job.log_path.read_text(encoding="utf-8", errors="replace"))


@app.get("/v1/jobs/{job_id}/metrics", summary="Structured metrics, if the job declares the 'metrics' capability")
def get_job_metrics(job_id: str, _: None = Depends(require_token)):
    """Reads output_dir/metrics.jsonl — one JSON object per line, written by
    the job's own script (e.g. {"step": 10, "loss": 0.3}). Any keys are
    allowed; this endpoint doesn't interpret them, it just returns the
    parsed records in order. Empty list if the job hasn't written any yet
    or doesn't declare 'metrics' in its capabilities."""
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    metrics_path = job.output_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    records = []
    for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


@app.get(
    "/v1/jobs/{job_id}/files",
    response_model=list[JobFileInfoExtended],
    summary="List files available in the job's output dir",
)
def list_job_files(job_id: str, _: None = Depends(require_token)):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    
    files_info = []
    for f in sorted(job.output_dir.iterdir()):
        if f.is_file():
            stat = f.stat()
            # Guess MIME type
            mime_type, _ = mimetypes.guess_type(str(f))
            if mime_type is None:
                # Fallback for common machine learning outputs
                if f.name.endswith(".jsonl"):
                    mime_type = "application/x-jsonlines"
                elif f.name.endswith(".pt") or f.name.endswith(".pth") or f.name.endswith(".ckpt"):
                    mime_type = "application/octet-stream"
                else:
                    mime_type = "text/plain"
            
            files_info.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "mime_type": mime_type,
                "modified_at": stat.st_mtime
            })
    return files_info


@app.get("/v1/jobs/{job_id}/files/{filename}", summary="Download a result file from the job's output dir")
def get_job_file(job_id: str, filename: str, _: None = Depends(require_token)):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    requested = (job.output_dir / filename).resolve()
    if job.output_dir.resolve() not in requested.parents or not requested.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(requested))


@app.delete("/v1/jobs/{job_id}", summary="Cancel a queued job or kill a running one")
def cancel_job(job_id: str, _: None = Depends(require_token)):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    
    # Check if job is already finished
    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(409, f"Cannot cancel job in '{job.status}' status. Job is already finished.")
        
    if not job_queue.cancel(job_id):
        raise HTTPException(500, "Failed to cancel the job")
        
    return {"cancelled": True}
