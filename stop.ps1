# Find by listening port instead of trusting a stored PID: the venv's
# python.exe launcher re-execs into a child process with a new PID, so the
# PID Start-Process reports is not reliably the one actually serving.
$conn = Get-NetTCPConnection -LocalPort 8077 -State Listen -ErrorAction SilentlyContinue
if (-not $conn) {
    Write-Host "Nothing listening on port 8077, server is probably not running."
    exit 0
}
$serverPid = $conn.OwningProcess
$children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$serverPid"
foreach ($child in $children) {
    Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue
}
Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
Remove-Item "$(Split-Path -Parent $MyInvocation.MyCommand.Path)\server.pid" -ErrorAction SilentlyContinue
Write-Host "Stopped server (pid $serverPid, plus any child interpreter process)."
