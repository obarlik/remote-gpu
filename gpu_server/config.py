import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GPU_SERVER_DATA_DIR", ROOT_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"
DATASETS_DIR = DATA_DIR / "datasets"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Python executable used to run training subprocesses.
# Defaults to the env where torch+CUDA was verified (miniforge3 base).
TRAIN_PYTHON_EXE = os.environ.get(
    "GPU_SERVER_TRAIN_PYTHON",
    str(Path.home() / "miniforge3" / "python.exe"),
)

AUTH_TOKEN = os.environ.get("GPU_SERVER_TOKEN")
