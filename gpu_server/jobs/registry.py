# Maps a task name to the module run via `python -m <module>` in a subprocess.
# Add new entries here to register new built-in training task types.
TASK_REGISTRY: dict[str, str] = {
    "transformer_train": "gpu_server.jobs.transformer_train",
}

# Generic escape hatch: run any user-supplied training script, not just the
# built-in tasks above. Keeps the server flexible for arbitrary training jobs
# (any framework: torch, OpenCL, sklearn, ...) while still using the same
# params.json / output-dir contract.
CUSTOM_SCRIPT_TASK = "custom_script"


def resolve_task_module(task: str) -> str:
    if task not in TASK_REGISTRY:
        known = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(f"Unknown task '{task}'. Known tasks: {known}, {CUSTOM_SCRIPT_TASK}")
    return TASK_REGISTRY[task]


def is_known_task(task: str) -> bool:
    return task in TASK_REGISTRY or task == CUSTOM_SCRIPT_TASK
