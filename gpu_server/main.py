import gzip
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from gpu_server.auth import require_token
from gpu_server.config import DATASETS_DIR as UPLOADS_DIR
from gpu_server.queue_manager import job_queue
from gpu_server.schemas import JobInfo, JobSubmitRequest

app = FastAPI(
    title="remote-gpu training server",
    description=(
        "Exposes this machine's GPU to remote clients for training jobs. "
        "All endpoints require `Authorization: Bearer <token>`. "
        "Submit a job with task='custom_script' to run any framework-agnostic "
        "training script (torch, OpenCL, raw CUDA kernels, ...), or task="
        "'transformer_train' for the built-in parametrized transformer LM."
    ),
    version="1.0.0",
)
# Compresses outgoing responses (job lists, logs, file downloads) when the
# client sends Accept-Encoding: gzip. Pure response-side, opt-in via that
# header, so clients that don't ask for it see no change at all.
app.add_middleware(GZipMiddleware, minimum_size=1024)


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
def upload_file(file: UploadFile, _: None = Depends(require_token)):
    """Generic upload endpoint: a dataset, a driver script, or a raw kernel
    source (.cl / .cu / .py) — anything a training job's script_path or
    params may need to reference by path. Returns {file_id, path}; pass
    `path` as script_path (or any custom param) when submitting a job.

    If the uploaded filename ends in '.gz', the body is treated as gzip and
    transparently decompressed to disk under the name with '.gz' stripped —
    this lets a slow/jittery link send fewer bytes without any other part
    of the contract (script_path, params, job execution) changing at all.
    Plain (non-.gz) uploads behave exactly as before."""
    file_id = uuid.uuid4().hex[:12]
    file_dir = UPLOADS_DIR / file_id
    file_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.bin").name  # strip any path components

    if safe_name.endswith(".gz"):
        dest = file_dir / safe_name[: -len(".gz")]
        with gzip.GzipFile(fileobj=file.file, mode="rb") as gz_in, open(dest, "wb") as out:
            shutil.copyfileobj(gz_in, out)
    else:
        dest = file_dir / safe_name
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)

    return {"file_id": file_id, "path": str(dest)}


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
