from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class RetentionPolicy(BaseModel):
    on_success: Literal["keep_all", "delete_artifacts", "delete_all"] = "keep_all"
    on_failure: Literal["keep_all", "delete_artifacts", "delete_all"] = "keep_all"
    on_cancelled: Literal["keep_all", "delete_artifacts", "delete_all"] = "keep_all"
    ttl_hours: int | None = Field(default=None, description="Time to live in hours after job completion/failure/cancellation.")


class JobSubmitRequest(BaseModel):
    task: str = Field(
        ...,
        description="'transformer_train' (built-in) or 'custom_script' (any uploaded script)",
        examples=["custom_script"],
    )
    label: str | None = Field(
        default=None,
        description=(
            "Free-form, human-readable description of what this run actually is — "
            "you decide the wording, the server never infers it from task/params. "
            "Shown in job listings and the dashboard instead of just the task name, "
            "so e.g. 'scaleup d640 attempt 3' is distinguishable from 'quick sanity check'."
        ),
        examples=["runo scaleup d640 attempt 3"],
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key/value pairs forwarded as JSON to the job's --params file. "
            "For 'custom_script', must include 'script_path' from a prior POST /v1/files."
        ),
        examples=[{"script_path": "C:/.../driver.py", "steps": 200}],
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "What this job's script actually provides, declared by you — the server "
            "never guesses. 'metrics' means it writes structured records to "
            "output_dir/metrics.jsonl (one JSON object per line, e.g. "
            "{\"step\": 10, \"loss\": 0.3}); the dashboard charts those directly instead "
            "of falling back to best-effort log parsing. 'resume' means the script reads "
            "a 'resume_from' key from its params (a checkpoint path) and continues "
            "training from it if present — this is the fixed convention, the same way "
            "'metrics' always means output_dir/metrics.jsonl."
        ),
        examples=[["metrics"]],
    )
    retention: RetentionPolicy | None = Field(
        default=None,
        description="Optional retention policy for the job's files and directory.",
    )


class UploadInitRequest(BaseModel):
    filename: str = Field(..., description="Target filename to store the file under.", examples=["dataset.bin"])
    gzip_encoded: bool = Field(
        default=False,
        description=(
            "True if the bytes you PUT are gzip-compressed as a transport encoding "
            "(decompressed on complete). Independent of the filename — a real .gz "
            "file you want stored as-is must leave this False."
        ),
    )


class JobInfo(BaseModel):
    id: str
    task: str
    label: str | None = None
    project: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    status: JobStatus
    params: dict[str, Any]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    output_dir: str
    retention: RetentionPolicy | None = None


class JobMoveRequest(BaseModel):
    project: str | None = Field(
        ...,
        description="Project name to assign this job to, or null to unassign it. Works for jobs in any state.",
        examples=["runo"],
    )


class TemplateCreateRequest(BaseModel):
    name: str = Field(..., examples=["runo_transformer_70k"])
    task: str = Field(..., examples=["custom_script"])
    defaults: dict[str, Any] = Field(default_factory=dict, examples=[{"steps": 70000, "lr": 3e-4}])
    required_params: list[str] = Field(
        default_factory=list,
        description="Param keys that must be present (from defaults or job-time overrides) before a job can start.",
        examples=[["script_path", "corpus_path"]],
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="What jobs from this template provide, e.g. ['metrics'] if the script writes metrics.jsonl.",
        examples=[["metrics"]],
    )


class TemplateUpdateRequest(BaseModel):
    defaults: dict[str, Any] | None = None
    required_params: list[str] | None = None
    capabilities: list[str] | None = None


class TemplateInfo(BaseModel):
    name: str
    task: str
    defaults: dict[str, Any]
    required_params: list[str]
    capabilities: list[str] = Field(default_factory=list)
    version: int
    created_at: float
    updated_at: float


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., examples=["runo"])
    template: str | None = Field(default=None, description="Snapshot a template's task/defaults at creation time.")
    task: str | None = Field(default=None, description="Required if no template is given.")
    defaults: dict[str, Any] = Field(
        default_factory=dict,
        description="Project-specific values; override the template's snapshot on matching keys.",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Used only when no template is given; otherwise snapshotted from the template.",
    )


class ProjectUpdateRequest(BaseModel):
    defaults: dict[str, Any] = Field(
        default_factory=dict,
        description="Partial update — only the given keys are added/changed, the rest of defaults is untouched.",
    )


class ProjectInfo(BaseModel):
    name: str
    template: str | None
    task: str
    defaults: dict[str, Any]
    required_params: list[str]
    capabilities: list[str] = Field(default_factory=list)
    created_at: float
    updated_at: float


class ProjectJobRequest(BaseModel):
    label: str | None = Field(
        default=None,
        description="Free-form description of this specific run, shown in job listings and the dashboard.",
        examples=["scaleup d640 attempt 3"],
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="This run's deltas only — merged on top of the project's defaults.",
    )


class JobLogsResponse(BaseModel):
    lines: list[str] = Field(..., description="List of log lines in the requested chunk.")
    next_cursor: int | None = Field(
        None,
        description="Byte offset to use as 'cursor' in the next request to fetch older logs. Null if no more logs.",
    )
    total_bytes: int = Field(..., description="Total size of the log file in bytes.")
    has_more: bool = Field(..., description="True if there are older log lines available before this chunk.")


class GPUStatusResponse(BaseModel):
    name: str = Field(..., description="GPU model name.")
    memory_total_mb: int = Field(..., description="Total VRAM in MB.")
    memory_used_mb: int = Field(..., description="Used VRAM in MB.")
    utilization_pct: int = Field(..., description="GPU core utilization percent.")
    temperature_c: int | None = Field(None, description="GPU temperature in Celsius (null if unavailable).")
    fan_speed_pct: int | None = Field(None, description="GPU fan speed percent (null if unavailable).")
    power_draw_w: float | None = Field(None, description="Current power draw in Watts (null if unavailable).")
    power_limit_w: float | None = Field(None, description="GPU power limit in Watts (null if unavailable).")
    
    # New Server System Status Fields
    cpu_utilization_pct: int = Field(0, description="CPU usage percentage.")
    ram_total_gb: float = Field(0.0, description="Total system RAM in GB.")
    ram_used_gb: float = Field(0.0, description="Used system RAM in GB.")
    net_download_kbps: float = Field(0.0, description="Network download speed in KB/s.")
    net_upload_kbps: float = Field(0.0, description="Network upload speed in KB/s.")
    disk_read_kbps: float = Field(0.0, description="Disk read speed in KB/s.")
    disk_write_kbps: float = Field(0.0, description="Disk write speed in KB/s.")
    cpu_temperature_c: int | None = Field(None, description="CPU temperature in Celsius (null if unavailable).")
    board_temperature_c: int | None = Field(None, description="Motherboard temperature in Celsius (null if unavailable).")
    
    # New Server Specs Fields
    cpu_model: str = Field("Unknown CPU", description="CPU processor model name.")
    cpu_cores: int = Field(1, description="Number of CPU cores.")
    gpu_driver_version: str = Field("Unknown", description="NVIDIA Driver version.")
    gpu_cuda_version: str = Field("Unknown", description="NVIDIA CUDA version.")
    torch_version: str = Field("Not Installed", description="Installed PyTorch version.")
    torch_cuda_available: bool = Field(False, description="Whether PyTorch CUDA device is available.")
    
    # Network Addresses
    intranet_ips: list[str] = Field(default_factory=list, description="List of intranet IP addresses.")
    tailscale_ips: list[str] = Field(default_factory=list, description="List of Tailscale VPN IP addresses.")
    fqdn: str = Field("Unknown", description="Fully Qualified Domain Name of the host.")


class JobFileInfoExtended(BaseModel):
    filename: str = Field(..., description="Name of the file.")
    size_bytes: int = Field(..., description="File size in bytes.")
    mime_type: str = Field(..., description="Detected MIME type of the file for rendering previews.")
    modified_at: float = Field(..., description="Last modification timestamp.")

