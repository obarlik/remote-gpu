#!/bin/bash
# Find the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PID=""
if [ -f server.pid ]; then
    PID=$(cat server.pid)
fi

if [ -z "$PID" ]; then
    # Fallback: find process matching the uvicorn app command line
    PID=$(pgrep -f "gpu_server.main:app")
fi

if [ -z "$PID" ]; then
    echo "Uvicorn server is not running (no PID found)."
    exit 0
fi

# Kill the process and any of its children
pkill -P $PID 2>/dev/null
kill -9 $PID 2>/dev/null
rm -f server.pid

echo "Stopped server (PID $PID and any child training processes)."
