# remote-gpu training server

A small FastAPI server that exposes this machine's RTX 4090 to remote clients
for training jobs over HTTP, with token auth and a sequential job queue
(one job runs at a time, GPU isn't shared between jobs).

## Run

Put the token in `.env` (gitignored): `GPU_SERVER_TOKEN=<long random token>`.

```powershell
.\start.ps1   # starts the server in the background, logs to server_stdout.log/server_stderr.log
.\stop.ps1    # stops it (finds the process by the port it's listening on, kills it and its child)
```

`GPU_SERVER_TRAIN_PYTHON` controls which Python runs training subprocesses
(defaults to `~/miniforge3/python.exe`, which already has torch+CUDA).
Expose port 8077 to the remote client via your VPN/tunnel of choice
(e.g. Tailscale) — do not expose it directly to the internet.

## Job model

Two ways to run a job, both via `POST /v1/jobs` with `{"task": ..., "params": {...}}`:

1. **Built-in task** — currently `transformer_train` (small causal transformer
   LM, fully configurable: `d_model`, `n_heads`, `n_layers`, `seq_len`,
   `batch_size`, `steps`, `lr`, `dtype` (`fp16`/`bf16`/`fp32`), `dataset_path`).
2. **`custom_script`** — run *any* training script, any framework (torch,
   OpenCL/pyopencl, sklearn, raw CUDA kernels via cupy, ...). Upload the
   script (and any kernel source files it needs) via `POST /v1/files`, then
   submit `{"task": "custom_script", "params": {"script_path": "<uploaded path>", ...}}`.
   The script just needs to accept `--params <json file> --output-dir <dir>`
   and write any output files into `output_dir`.

This is the escape hatch that makes the server framework-agnostic — a script
that loads a raw `.cl` kernel file and runs it via pyopencl works exactly
the same way as a pure-torch script.

## Dashboard

`GET /dashboard` — a small browser page (no build step, vanilla JS) showing
the job list, live status, and a best-effort loss chart parsed from
whatever "step N ... loss X" or "loss: X" text appears in a job's log.
It polls the same `/v1/*` API every 3s; enter the bearer token once and
it's kept in the browser's localStorage. Since it's just a thin client over
the existing API, it doesn't affect running jobs and exposes nothing the
API doesn't already expose.

## Endpoints

- `GET /v1/gpu` — name, VRAM, utilization (via `nvidia-smi`)
- `POST /v1/files` — upload a dataset, driver script, or kernel source in
  one shot. `?gzip_encoded=true` decompresses the body on arrival — use
  this only when the bytes you're sending are gzip as a *transport*
  encoding; a real `.gz` dataset you want stored byte-for-byte must leave
  this unset (the filename alone never implies decompression).
- `POST /v1/uploads` / `PUT /v1/uploads/{id}?offset=N` / `POST
  /v1/uploads/{id}/complete` — resumable, chunkable upload. Send the file
  as one PUT or many, in order; if the connection drops, `GET
  /v1/uploads/{id}` tells you the received_bytes offset to resume from.
  Wrong offset is rejected with 409 instead of corrupting the file.
  `DELETE /v1/uploads/{id}` aborts and discards. Same `gzip_encoded` flag
  as `/v1/files`, set at session start.
- `POST /v1/jobs` — submit a job, returns job id
- `GET /v1/jobs` / `GET /v1/jobs/{id}` — list / inspect status
- `GET /v1/jobs/{id}/logs` — full stdout/stderr so far
- `GET /v1/jobs/{id}/files` — list output files a job produced (name + size)
- `GET /v1/jobs/{id}/files/{filename}` — download a result file (e.g.
  checkpoint). Supports `Range` requests, so a dropped download can resume
  instead of restarting.
- `DELETE /v1/jobs/{id}` — cancel a queued or running job

All endpoints require `Authorization: Bearer <GPU_SERVER_TOKEN>`.

Responses are gzip-compressed automatically when the client sends
`Accept-Encoding: gzip` (clients that don't send it see no change).

## Example: custom kernel-based job

```bash
curl -F file=@my_matmul_kernel.cl -H "Authorization: Bearer $TOKEN" http://host:8077/v1/files
curl -F file=@driver.py -H "Authorization: Bearer $TOKEN" http://host:8077/v1/files

curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"task":"custom_script","params":{"script_path":"<driver.py path>","kernel_path":"<kernel.cl path>"}}' \
  http://host:8077/v1/jobs
```

`driver.py` reads `--params` (a JSON file containing `kernel_path` plus
whatever else you passed), compiles the kernel itself, and runs the
training loop — the server never needs to know what's inside it.
