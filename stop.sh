#!/bin/bash
# Find the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Find process listening on port 8077
PID=$(lsof -t -i:8077 2>/dev/null)
if [ -z "$PID" ]; then
    # Fallback to reading server.pid
    if [ -f server.pid ]; then
        PID=$(cat server.pid)
    fi
fi

if [ -z "$PID" ]; then
    echo "Nothing listening on port 8077, and server.pid not found."
    exit 0
fi

# Kill the process and any of its children
pkill -P $PID 2>/dev/null
kill -9 $PID 2>/dev/null
rm -f server.pid

echo "Stopped server (PID $PID and any child training processes)."
