"""Templates and projects: a lightweight experiment-tracking layer on top
of the job queue. Template = reusable blueprint (data, not code). Project =
durable instance (snapshotted from a template, optionally) that jobs run
against by specifying only this run's deltas."""
from fastapi import APIRouter, Depends, HTTPException

from gpu_server.auth import require_token
from gpu_server.projects import project_manager
from gpu_server.queue_manager import job_queue
from gpu_server.schemas import (
    JobInfo,
    ProjectCreateRequest,
    ProjectInfo,
    ProjectJobRequest,
    ProjectUpdateRequest,
    TemplateCreateRequest,
    TemplateInfo,
    TemplateUpdateRequest,
)
from gpu_server.templates import template_manager

router = APIRouter(dependencies=[Depends(require_token)])


@router.post("/v1/templates", response_model=TemplateInfo, summary="Define a reusable job blueprint")
def create_template(req: TemplateCreateRequest):
    """Templates are pure data — task + defaults + which keys are required —
    not hardcoded job types. Any task this server supports can be wrapped
    in a template."""
    try:
        return template_manager.create(req.name, req.task, req.defaults, req.required_params, req.capabilities)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/v1/templates", response_model=list[TemplateInfo], summary="List all templates")
def list_templates():
    return template_manager.list()


@router.get("/v1/templates/{name}", response_model=TemplateInfo, summary="Get one template")
def get_template(name: str):
    tpl = template_manager.get(name)
    if tpl is None:
        raise HTTPException(404, "Template not found")
    return tpl


@router.patch("/v1/templates/{name}", response_model=TemplateInfo, summary="Update a template's defaults")
def update_template(name: str, req: TemplateUpdateRequest):
    """Only existing projects created from this template are unaffected —
    they snapshotted the old values at creation time. Only new projects
    pick up this change."""
    try:
        return template_manager.update(name, req.defaults, req.required_params, req.capabilities)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/v1/templates/{name}", summary="Delete a template")
def delete_template(name: str):
    if not template_manager.delete(name):
        raise HTTPException(404, "Template not found")
    return {"deleted": True}


@router.post("/v1/projects", response_model=ProjectInfo, summary="Create a project (optionally from a template)")
def create_project(req: ProjectCreateRequest):
    """A project holds durable defaults (file paths, hyperparams) so later
    job submissions only need to specify what's different this run."""
    try:
        return project_manager.create(req.name, req.template, req.task, req.defaults, req.capabilities)
    except ValueError as exc:
        raise HTTPException(409 if "already exists" in str(exc) else 400, str(exc)) from exc


@router.get("/v1/projects", response_model=list[ProjectInfo], summary="List all projects")
def list_projects():
    return project_manager.list()


@router.get("/v1/projects/{name}", response_model=ProjectInfo, summary="Get one project")
def get_project(name: str):
    project = project_manager.get(name)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@router.patch("/v1/projects/{name}", response_model=ProjectInfo, summary="Partially update a project's defaults")
def update_project(name: str, req: ProjectUpdateRequest):
    """Only the given keys change — e.g. {"corpus_path": "<new path>"} after
    uploading a new dataset version, without re-stating everything else."""
    try:
        return project_manager.update_defaults(name, req.defaults)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/v1/projects/{name}", summary="Delete a project (its past jobs/files are untouched)")
def delete_project(name: str):
    if not project_manager.delete(name):
        raise HTTPException(404, "Project not found")
    return {"deleted": True}


@router.post("/v1/projects/{name}/jobs", response_model=JobInfo, summary="Submit a job with only this run's deltas")
def submit_project_job(name: str, req: ProjectJobRequest):
    """Merges the project's defaults with these overrides and validates
    required_params before queueing — catches missing config early instead
    of failing deep inside the training script."""
    try:
        task, final_params, capabilities = project_manager.resolve_job(name, req.params)
        job = job_queue.submit(task, final_params, project=name, capabilities=capabilities)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return job.to_dict()


@router.get("/v1/projects/{name}/jobs", response_model=list[JobInfo], summary="List all job runs under a project")
def list_project_jobs(name: str):
    if project_manager.get(name) is None:
        raise HTTPException(404, "Project not found")
    return [job.to_dict() for job in job_queue.list_jobs_by_project(name)]
