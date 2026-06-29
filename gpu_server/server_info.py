"""Single source of truth for the server's version and feature flags, so
clients can discover what's supported via GET /v1/server-info instead of
guessing from a version number or re-reading the docs after every change."""

VERSION = "1.6.0"

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
    "job_retention",
    "interactive_terminal",
    "help_endpoint",
]

WHATSNEW = {
    "1.6.0": [
        "Added automated job retention policies (TTL and completion-based artifact deletion) to prevent disk space bloat.",
        "Added interactive virtual terminal support using Linux PTYs and WebSockets for real-time tqdm and console input/output.",
        "Added partitioned history metadata store to eliminate disk load/save overhead.",
        "Added console-friendly Markdown API guide endpoint via GET /v1/help."
    ],
    "1.5.0": [
        "Implemented ultra-compact Live Stats double-column grid layout in GPU Monitor.",
        "Relocated GPU monitor to the sidebar and restricted panel heights to prevent layout scrolling.",
        "Added CPU and Motherboard temperature sensors to System Health."
    ],
    "1.4.0": [
        "Implemented full server system resource monitoring (CPU, RAM, Net, Disk).",
        "Added detailed specs tab showing Host GPU Driver/CUDA, PyTorch config, and CPU details."
    ],
    "1.3.0": [
        "Overhauled dashboard design with a unified glassmorphic single flow layout.",
        "Added interactive TV/Focus mode modal with auto-scaling metrics charts.",
        "Added collapsible save key badge and nowrap compact jobs list layout."
    ],
    "1.2.0": [
        "Added customized minimalist dark-themed scrollbars across all panels.",
        "Added log pagination support to load older logs backwards on scroll.",
        "Added enhanced GPU monitoring and client-side artifact browser."
    ],
    "1.1.0": [
        "Added custom script task execution support to run framework-agnostic training scripts.",
        "Added templates and projects API endpoints with parameter snapshotting.",
        "Added file-sharing capabilities via GET /v1/projects/{name}/files."
    ],
    "1.0.0": [
        "Initial release with sequential training job queue.",
        "Built-in transformer LM training task ('transformer_train') fully configurable.",
        "Added token authorization and basic web dashboard with loss charts."
    ]
}
