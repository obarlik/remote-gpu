#!/bin/bash
# Find the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Load env variables if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set the Python environment used to run training subprocesses
# to our Linux Python virtual environment.
export GPU_SERVER_TRAIN_PYTHON="$DIR/.venv-linux/bin/python"

# Start the FastAPI server in the background and redirect output
nohup "$DIR/.venv-linux/bin/python" -m uvicorn gpu_server.main:app --host 0.0.0.0 --port 8077 > server_stdout.log 2> server_stderr.log &
PID=$!
echo $PID > server.pid
echo "Server started in background with PID $PID. Logs: server_stdout.log / server_stderr.log"
