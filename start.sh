#!/bin/bash
# Find the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Load env variables if .env exists (using absolute path)
if [ -f "$DIR/.env" ]; then
    export $(grep -v '^#' "$DIR/.env" | xargs)
fi

# Set the Python environment used to run training subprocesses
# to our Linux Python virtual environment.
export GPU_SERVER_TRAIN_PYTHON="$DIR/.venv-linux/bin/python"

# Check if server is already running
if [ -f server.pid ]; then
    OLD_PID=$(cat server.pid)
    if kill -0 $OLD_PID 2>/dev/null; then
        echo "Server is already running with PID $OLD_PID. Stop it first using ./stop.sh"
        exit 1
    fi
    rm -f server.pid
fi

# Start the FastAPI server in the background and redirect output
nohup "$DIR/.venv-linux/bin/python" -m uvicorn gpu_server.main:app --host 0.0.0.0 --port 8077 > server_stdout.log 2> server_stderr.log &
PID=$!
echo $PID > server.pid

echo "Server process launched (PID: $PID). Waiting for port 8077 to accept connections..."

# Poll the port until it is open or timeout at 15 seconds
TIMEOUT=15
ELAPSED=0
PORT_OPEN=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    if bash -c 'cat < /dev/null > /dev/tcp/127.0.0.1/8077' 2>/dev/null; then
        PORT_OPEN=1
        break
    fi
    sleep 0.5
    ELAPSED=$((ELAPSED + 1))
done

if [ $PORT_OPEN -eq 1 ]; then
    echo "Server successfully started and listening on port 8077."
    echo "Uvicorn running on http://127.0.0.1:8077 (Press CTRL+C to quit)"
else
    echo "Warning: Server was launched but port 8077 did not become active within 15 seconds."
    echo "Check server_stderr.log for details."
fi
