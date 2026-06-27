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
    status: JobStatus
    params: dict[str, Any]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    output_dir: str
