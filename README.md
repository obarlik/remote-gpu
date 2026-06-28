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

`task` itself is not a free-form label — it's either `custom_script` or a
name registered in source (`transformer_train`), since it decides how the
server launches the process. To describe what a specific run actually
*is*, set `"label"` (e.g. `"scaleup d640 attempt 3"`) on submission — free
text, you choose the wording, shown in job listings and the dashboard
instead of everyone just seeing `custom_script`. Works the same way on
`POST /v1/jobs` and `POST /v1/projects/{name}/jobs`.

## Templates and projects

For recurring job shapes, avoid re-sending everything on every submission:

- `POST /v1/templates` — define a reusable blueprint: `name`, `task`,
  `defaults`, and `required_params` (keys that must be present before a job
  can start — catches missing config early). Pure data, not hardcoded.
- `POST /v1/projects` — create a project, optionally `"template": "<name>"`.
  The template's `defaults`/`required_params` are **snapshotted** at
  creation time — editing the template later never changes existing
  projects, only new ones.
- `PATCH /v1/projects/{name}` — update just the given keys in `defaults`
  (e.g. point at a newly uploaded dataset without re-stating hyperparams).
- `POST /v1/projects/{name}/jobs` — submit a run with only this run's
  deltas (`{"params": {...}}`); merged with the project's defaults, with
  `required_params` validated before queueing.
- `GET /v1/projects/{name}/jobs` — every run submitted under this project.
- `GET /v1/projects/{name}/files` — which of the project's `defaults` are
  actual files on disk (path + size), e.g. `corpus_path`/`script_path` —
  not plain hyperparams like `steps`/`lr`. These are exactly what every
  job under the project inherits unless overridden.
- `PATCH /v1/jobs/{id}` — assign, move, or unassign any job's project after
  the fact (`{"project": "<name>" | null}`), even one that's still running
  or was submitted without a project. Safe at any time: the project field
  is just a label, the actual training process never reads it.

Templates, projects, and job history all persist to JSON files under
`data/` and survive a server restart.

**File scope is per-job, never global.** There's no "latest artifact"
registry and no automatic sharing between jobs — a job's outputs live only
in its own `output_dir`, found via `GET /v1/jobs/{id}/files`. If a new job
should use a previous job's output (a freshly built corpus, a checkpoint,
...), you carry the path forward explicitly: either pass it directly in
`params`, or — for a recurring project — `PATCH /v1/projects/{name}` with
that absolute path in `defaults` so every future run under that project
picks it up automatically. Nothing happens implicitly — but once a file is
shared at the project level this way, it *is* visible to that project's
jobs (and to you, via `GET /v1/projects/{name}/files`) without re-stating
it on every submission.

## Capabilities and structured metrics

A job (or its template/project) can declare `"capabilities": ["metrics"]`.
This is purely declarative — the server never guesses what a script does:

- If declared, the script is expected to write `output_dir/metrics.jsonl`
  (one JSON object per line, any keys, e.g. `{"step": 10, "loss": 0.3}`).
  `GET /v1/jobs/{id}/metrics` returns the parsed records.
- The dashboard checks a job's `capabilities`: if `"metrics"` is present, it
  charts every numeric key found in the structured records as its own line
  with a legend. Otherwise it falls back to a best-effort regex over the
  raw log text for `"step N ... loss X"` / `"loss: X"`.
- `"resume"` is a fixed convention too: it declares that the script reads a
  `resume_from` key from its params (a checkpoint path) and continues
  training from it if present. There's no dedicated endpoint for this —
  resuming just means submitting a new job/project-run with
  `{"resume_from": "<checkpoint path>", ...}` in `params`. A script must
  actually implement reading `resume_from` for this to do anything; the
  capability declaration only tells callers it's safe to try.

## Dashboard

`GET /dashboard` — a small browser page (no build step, vanilla JS) showing
the job list, live status, and a best-effort loss chart parsed from
whatever "step N ... loss X" or "loss: X" text appears in a job's log.
It polls the same `/v1/*` API every 3s; enter the bearer token once and
it's kept in the browser's localStorage. Since it's just a thin client over
the existing API, it doesn't affect running jobs and exposes nothing the
API doesn't already expose.

## Endpoints

- `GET /v1/server-info` — version + a list of feature flags, so a client
  can check what's supported in one request. Flags are only ever appended,
  never renamed or removed, so checking for one by name is safe long-term.
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
- `PATCH /v1/jobs/{id}` — assign/move/unassign a job's project
  (`{"project": "<name>" | null}`); see "Templates and projects" above
- `GET /v1/jobs/{id}/logs` — full stdout/stderr so far
- `GET /v1/jobs/{id}/metrics` — structured records if the job declared the
  `metrics` capability; see "Capabilities and structured metrics" above
- `GET /v1/jobs/{id}/files` — list output files a job produced (name + size)
- `GET /v1/jobs/{id}/files/{filename}` — download a result file (e.g.
  checkpoint). Supports `Range` requests, so a dropped download can resume
  instead of restarting.
- `DELETE /v1/jobs/{id}` — cancel a queued or running job
- Template/project endpoints (`/v1/templates*`, `/v1/projects*`) are listed
  in "Templates and projects" above, not repeated here.

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
