$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Get-Content "$root\.env" | ForEach-Object {
    if ($_ -match '^(.+?)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] }
}
$proc = Start-Process -NoNewWindow -PassThru `
    -FilePath "$root\.venv-server\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "gpu_server.main:app", "--host", "0.0.0.0", "--port", "8077" `
    -RedirectStandardOutput "$root\server_stdout.log" `
    -RedirectStandardError "$root\server_stderr.log" `
    -WorkingDirectory $root
$proc.Id | Set-Content "$root\server.pid"
Write-Host "Server started, pid $($proc.Id). Logs: server_stdout.log / server_stderr.log"
