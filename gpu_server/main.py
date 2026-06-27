import gzip
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from gpu_server.auth import require_token
from gpu_server.config import DATASETS_DIR as UPLOADS_DIR
from gpu_server.queue_manager import job_queue
from gpu_server.schemas import JobInfo, JobSubmitRequest, UploadInitRequest
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
        "when the client sends `Accept-Encoding: gzip`."
    ),
    version="1.1.0",
)
# Compresses outgoing responses (job lists, logs, file downloads) when the
# client sends Accept-Encoding: gzip. Pure response-side, opt-in via that
# header, so clients that don't ask for it see no change at all.
app.add_middleware(GZipMiddleware, minimum_size=1024)


_DASHBOARD_HTML = (Path(__file__).resolve().parent / "static" / "dashboard.html").read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse, summary="Web UI: job list, status, and a best-effort loss chart")
def dashboard():
    """Static page; it calls the same /v1/* API from the browser using a
    bearer token you enter once (stored in the browser's localStorage).
    Doesn't expose anything not already available via the API."""
    return _DASHBOARD_HTML


@app.get("/v1/gpu", summary="Current GPU name, VRAM, and utilization")
def gpu_status(_: None = Depends(require_token)):
    """Queries nvidia-smi for a live snapshot of the GPU this server controls."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        name, mem_total, mem_used, util = (x.strip() for x in out.strip().split(","))
        return {
            "name": name,
            "memory_total_mb": int(mem_total),
            "memory_used_mb": int(mem_used),
            "utilization_pct": int(util),
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
        job = job_queue.submit(req.task, req.params)
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


@app.get("/v1/jobs/{job_id}/logs", response_class=PlainTextResponse, summary="Job stdout/stderr so far")
def get_job_logs(job_id: str, _: None = Depends(require_token)):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if not job.log_path.exists():
        return ""
    return job.log_path.read_text(encoding="utf-8", errors="replace")


@app.get("/v1/jobs/{job_id}/files", summary="List files available in the job's output dir")
def list_job_files(job_id: str, _: None = Depends(require_token)):
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return [
        {"filename": f.name, "size_bytes": f.stat().st_size}
        for f in sorted(job.output_dir.iterdir())
        if f.is_file()
    ]


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
    if not job_queue.cancel(job_id):
        raise HTTPException(404, "Job not found or already finished")
    return {"cancelled": True}
