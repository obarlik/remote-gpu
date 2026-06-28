#!/bin/bash
# Find the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PID=""
if [ -f server.pid ]; then
    PID=$(cat server.pid)
fi

# Fallback/Additional: pgrep to find all uvicorn processes under this path
ALL_PIDS=$(pgrep -f "gpu_server.main:app")

if [ -z "$PID" ] && [ -z "$ALL_PIDS" ]; then
    echo "Uvicorn server is not running (no PID found)."
    rm -f server.pid
    exit 0
fi

# Target main PID first
if [ -n "$PID" ] && kill -0 $PID 2>/dev/null; then
    pkill -P $PID 2>/dev/null
    kill -9 $PID 2>/dev/null
fi

# Clean up other uvicorn process matches if any
for p in $ALL_PIDS; do
    if kill -0 $p 2>/dev/null; then
        pkill -P $p 2>/dev/null
        kill -9 $p 2>/dev/null
    fi
done

rm -f server.pid
echo "Stopped server processes and child training processes."
