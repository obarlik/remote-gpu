from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class JobSubmitRequest(BaseModel):
    task: str = Field(
        ...,
        description="'transformer_train' (built-in) or 'custom_script' (any uploaded script)",
        examples=["custom_script"],
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
    project: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    status: JobStatus
    params: dict[str, Any]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    output_dir: str


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
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="This run's deltas only — merged on top of the project's defaults.",
    )
