"""Single source of truth for the server's version and feature flags, so
clients can discover what's supported via GET /v1/server-info instead of
guessing from a version number or re-reading the docs after every change."""

VERSION = "1.5.0"

FEATURES = [
    "custom_script_task",
    "transformer_train_task",
    "resumable_chunked_uploads",
    "gzip_transport_encoding",
    "gzip_response_compression",
    "range_resumable_downloads",
    "templates",
    "projects",
    "project_job_deltas",
    "job_project_reassignment",
    "declared_capabilities",
    "structured_metrics",
    "resume_from_convention",
    "job_labels",
    "dashboard_ui",
    "server_info_endpoint",
    "log_pagination",
    "enhanced_gpu_monitoring",
    "artifact_browser",
]
