"""Per-user research workspace: projects, stored datasets and saved analysis results."""
import io
import json
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from .auth import get_current_user
from .database import get_db
from . import dataprep, methods_text
from .project_report import build_project_report
from .reports import DOCX_MEDIA_TYPE, build_report
from .models import AnalysisResult, Dataset, PrepRecipe, Project, ProjectMember, User
from .modules.common import read_uploaded_table
from .project_access import EDITOR, OWNER, VIEWER, accessible_project_ids, can_change, item_for, project_for

router = APIRouter(prefix="/workspace", tags=["Workspace"])

MAX_DATASET_BYTES = 10 * 1024 * 1024        # per uploaded file
MAX_RESULT_BYTES = 5 * 1024 * 1024          # per saved analysis (results can include chart images)
MAX_USER_STORAGE_BYTES = 100 * 1024 * 1024  # datasets + saved results + recipes, per user
MAX_RECIPE_BYTES = 256 * 1024              # a recipe is a list of settings, never data
MAX_RECIPE_STEPS = 50
PROJECT_STATUSES = ("Empty", "Ready", "In Progress", "Completed")

SAMPLE_PROJECT = {
    "name": "Dummy project",
    "status": "Ready",
    "samples": 120,
    "variables": 6,
    "test_used": "One-Way ANOVA, LSD Test",
    "tasks": [
        {"id": 101, "text": "Add your CSV dataset file", "completed": True},
        {"id": 102, "text": "Verify variable columns in data editor", "completed": True},
        {"id": 103, "text": "Run ANOVA with Duncan / LSD post-hoc", "completed": False},
        {"id": 104, "text": "Download high-resolution 300 DPI chart", "completed": False},
    ],
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Task(BaseModel):
    id: int
    text: str = Field(min_length=1, max_length=500)
    completed: bool = False


def _check_status(value):
    if value is not None and value not in PROJECT_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(PROJECT_STATUSES)}")
    return value


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    status: str = "Empty"
    samples: int = Field(0, ge=0)
    variables: int = Field(0, ge=0)
    test_used: str = Field("None", max_length=200)
    tasks: List[Task] = []

    _status = field_validator("status")(_check_status)


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    status: Optional[str] = None
    samples: Optional[int] = Field(None, ge=0)
    variables: Optional[int] = Field(None, ge=0)
    test_used: Optional[str] = Field(None, max_length=200)
    tasks: Optional[List[Task]] = None

    _status = field_validator("status")(_check_status)


class ResultIn(BaseModel):
    module: str = Field(min_length=1, max_length=50)
    title: str = Field(min_length=1, max_length=200)
    parameters: dict = {}
    result: dict
    dataset_id: Optional[int] = None
    report: Optional[dict] = None  # the module's Word-report content, reused in project reports


class RecipeIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    source_dataset: Optional[str] = Field(None, max_length=255)
    steps: List[dict] = Field(min_length=1, max_length=MAX_RECIPE_STEPS)

    @field_validator("steps")
    @classmethod
    def _known_steps(cls, steps):
        for position, step in enumerate(steps, start=1):
            if step.get("type") not in dataprep.STEPS:
                raise ValueError(f"step {position} has an unknown type {step.get('type')!r}")
        return steps


class ProjectReportIn(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    result_ids: Optional[List[int]] = None  # default: every saved analysis in the project
    recipe_ids: Optional[List[int]] = None  # default: every recipe in the project
    include_activity: bool = True
    include_references: bool = True


class ImportIn(BaseModel):
    projects: List[dict]  # projects as previously stored in the browser (localStorage)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _person(user: Optional[User]):
    if user is None:
        return None
    return {"id": user.id, "full_name": user.full_name, "email": user.email}


def _people(db: Session, ids):
    ids = {i for i in ids if i is not None}
    return {u.id: u for u in db.query(User).filter(User.id.in_(ids)).all()} if ids else {}


def _storage_used(db: Session, user: User) -> int:
    datasets = db.query(func.coalesce(func.sum(Dataset.size_bytes), 0)).filter(Dataset.user_id == user.id).scalar()
    results = db.query(func.coalesce(func.sum(AnalysisResult.size_bytes), 0)).filter(AnalysisResult.user_id == user.id).scalar()
    recipes = db.query(func.coalesce(func.sum(PrepRecipe.size_bytes), 0)).filter(PrepRecipe.user_id == user.id).scalar()
    return int(datasets) + int(results) + int(recipes)


def _require_quota(db: Session, user: User, extra_bytes: int):
    used = _storage_used(db, user)
    if used + extra_bytes > MAX_USER_STORAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Storage limit reached ({MAX_USER_STORAGE_BYTES // (1024 * 1024)} MB). Delete datasets or saved results to free space."
        )


def _iso(value: Optional[datetime]):
    return value.isoformat() if value else None


def _project_json(project: Project, dataset_count: int = 0, result_count: int = 0, role: str = OWNER,
                  owner: Optional[User] = None, member_count: int = 0):
    return {
        "role": role,
        "owner": _person(owner),
        "member_count": member_count,
        "id": project.id,
        "name": project.name,
        "status": project.status,
        "samples": project.samples,
        "variables": project.variables,
        "test_used": project.test_used,
        "tasks": project.tasks or [],
        "dataset_count": dataset_count,
        "result_count": result_count,
        "created_at": _iso(project.created_at),
        "updated_at": _iso(project.updated_at),
    }


def _dataset_json(dataset: Dataset):
    return {
        "id": dataset.id,
        "project_id": dataset.project_id,
        "filename": dataset.filename,
        "content_type": dataset.content_type,
        "size_bytes": dataset.size_bytes,
        "n_rows": dataset.n_rows,
        "n_columns": dataset.n_columns,
        "columns": dataset.columns or [],
        "created_at": _iso(dataset.created_at),
    }


def _result_json(result: AnalysisResult, include_payload: bool = False):
    data = {
        "id": result.id,
        "project_id": result.project_id,
        "dataset_id": result.dataset_id,
        "module": result.module,
        "title": result.title,
        "parameters": result.parameters or {},
        "size_bytes": result.size_bytes,
        "created_at": _iso(result.created_at),
    }
    if include_payload:
        data["result"] = result.result
    return data


def _recipe_json(recipe: PrepRecipe):
    return {
        "id": recipe.id,
        "project_id": recipe.project_id,
        "name": recipe.name,
        "description": recipe.description,
        "source_dataset": recipe.source_dataset,
        "steps": recipe.steps or [],
        "step_count": len(recipe.steps or []),
        "size_bytes": recipe.size_bytes,
        "created_at": _iso(recipe.created_at),
        "updated_at": _iso(recipe.updated_at),
    }


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------
def _accepted_members(db: Session, project_ids):
    if not project_ids:
        return []
    return (db.query(ProjectMember)
            .filter(ProjectMember.project_id.in_(project_ids), ProjectMember.accepted_at.isnot(None)).all())


@router.get("/projects")
def list_projects(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Projects the user owns and projects shared with them, most recently changed first."""
    accessible = accessible_project_ids(db, current_user)
    projects = (db.query(Project).filter(Project.id.in_(accessible)).order_by(Project.updated_at.desc()).all()
                if accessible else [])
    if not projects:
        # A sample project shows new users how the workspace is organised
        sample = Project(user_id=current_user.id, **SAMPLE_PROJECT)
        db.add(sample)
        db.commit()
        db.refresh(sample)
        projects = [sample]

    ids = [p.id for p in projects]
    dataset_counts = dict(db.query(Dataset.project_id, func.count(Dataset.id)).filter(Dataset.project_id.in_(ids)).group_by(Dataset.project_id).all())
    result_counts = dict(db.query(AnalysisResult.project_id, func.count(AnalysisResult.id)).filter(AnalysisResult.project_id.in_(ids)).group_by(AnalysisResult.project_id).all())
    members = _accepted_members(db, ids)
    roles = {m.project_id: m.role for m in members if m.user_id == current_user.id}
    member_counts = {}
    for m in members:
        member_counts[m.project_id] = member_counts.get(m.project_id, 0) + 1
    owners = _people(db, [p.user_id for p in projects])
    return [_project_json(p, dataset_counts.get(p.id, 0), result_counts.get(p.id, 0),
                          role=OWNER if p.user_id == current_user.id else roles.get(p.id, "viewer"),
                          owner=owners.get(p.user_id), member_count=member_counts.get(p.id, 0))
            for p in projects]


@router.post("/projects", status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project = Project(user_id=current_user.id, **{**body.model_dump(), "tasks": [t.model_dump() for t in body.tasks]})
    db.add(project)
    db.commit()
    db.refresh(project)
    return _project_json(project, owner=current_user)


def member_json(member: ProjectMember, user: Optional[User] = None):
    return {
        "id": member.id,
        "email": member.email,
        "full_name": user.full_name if user else None,
        "role": member.role,
        "status": "active" if member.accepted_at else "invited",
        "invited_at": _iso(member.created_at),
        "accepted_at": _iso(member.accepted_at),
    }


@router.get("/projects/{project_id}")
def get_project(project_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project, role = project_for(db, project_id, current_user)
    datasets = db.query(Dataset).filter(Dataset.project_id == project.id).order_by(Dataset.created_at.desc()).all()
    results = db.query(AnalysisResult).filter(AnalysisResult.project_id == project.id).order_by(AnalysisResult.created_at.desc()).all()
    recipes = db.query(PrepRecipe).filter(PrepRecipe.project_id == project.id).order_by(PrepRecipe.updated_at.desc()).all()
    members = db.query(ProjectMember).filter(ProjectMember.project_id == project.id).order_by(ProjectMember.created_at).all()
    if role != OWNER:
        members = [m for m in members if m.accepted_at]  # pending invitations are the owner's business
    people = _people(db, [project.user_id] + [m.user_id for m in members]
                     + [i.user_id for i in (*datasets, *results, *recipes)])

    def with_author(item, data):
        return {**data, "added_by": _person(people.get(item.user_id)), "can_change": can_change(role, item, current_user)}

    data = _project_json(project, len(datasets), len(results), role=role, owner=people.get(project.user_id),
                         member_count=sum(1 for m in members if m.accepted_at))
    data["datasets"] = [with_author(d, _dataset_json(d)) for d in datasets]
    data["results"] = [with_author(r, _result_json(r)) for r in results]
    data["recipes"] = [with_author(r, _recipe_json(r)) for r in recipes]
    data["members"] = [member_json(m, people.get(m.user_id)) for m in members]
    return data


@router.patch("/projects/{project_id}")
def update_project(project_id: int, body: ProjectUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project, role = project_for(db, project_id, current_user, EDITOR)
    changes = body.model_dump(exclude_unset=True)
    if role != OWNER and "name" in changes and changes["name"] != project.name:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the project owner can rename the project.")
    if "tasks" in changes and body.tasks is not None:
        changes["tasks"] = [t.model_dump() for t in body.tasks]
    for field, value in changes.items():
        if value is not None:
            setattr(project, field, value)
    project.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(project)
    return _project_json(project, role=role, owner=db.get(User, project.user_id))


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project, _ = project_for(db, project_id, current_user, OWNER)
    db.delete(project)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/import")
def import_browser_projects(body: ImportIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """One-time import of projects that were kept in the browser before accounts stored them."""
    imported = 0
    for item in body.projects:
        name = str(item.get("name") or "").strip()[:200]
        if not name or name == SAMPLE_PROJECT["name"]:
            continue
        tasks = []
        for task in item.get("tasks") or []:
            text = str(task.get("text") or "").strip()[:500]
            if text:
                tasks.append({"id": int(task.get("id") or len(tasks) + 1), "text": text, "completed": bool(task.get("completed"))})
        created = None
        try:
            created = datetime.strptime(str(item.get("created")), "%Y-%m-%d")
        except ValueError:
            pass
        project_status = item.get("status") if item.get("status") in PROJECT_STATUSES else "Empty"
        db.add(Project(
            user_id=current_user.id, name=name, status=project_status,
            samples=max(0, int(item.get("samples") or 0)), variables=max(0, int(item.get("variables") or 0)),
            test_used=str(item.get("testUsed") or "None")[:200], tasks=tasks,
            **({"created_at": created, "updated_at": created} if created else {}),
        ))
        imported += 1
    db.commit()
    return {"imported": imported}


@router.get("/storage")
def storage_summary(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return {
        "used_bytes": _storage_used(db, current_user),
        "limit_bytes": MAX_USER_STORAGE_BYTES,
        "max_dataset_bytes": MAX_DATASET_BYTES,
        "max_result_bytes": MAX_RESULT_BYTES,
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@router.post("/projects/{project_id}/datasets", status_code=status.HTTP_201_CREATED)
def upload_dataset(project_id: int, file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project, _ = project_for(db, project_id, current_user, EDITOR)
    file.file.seek(0)
    content = file.file.read()
    if len(content) > MAX_DATASET_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Datasets can be at most {MAX_DATASET_BYTES // (1024 * 1024)} MB."
        )
    _require_quota(db, current_user, len(content))
    df = read_uploaded_table(file, str_columns=True)  # validates that the file is a readable table

    dataset = Dataset(
        user_id=current_user.id, project_id=project.id, filename=(file.filename or "dataset.csv")[:255],
        content_type=(file.content_type or "application/octet-stream")[:120], size_bytes=len(content),
        n_rows=int(len(df)), n_columns=int(df.shape[1]), columns=list(df.columns), content=content,
    )
    db.add(dataset)
    if project.samples == 0 and project.variables == 0:
        project.samples, project.variables = int(len(df)), int(df.shape[1])
    project.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(dataset)
    return _dataset_json(dataset)


@router.get("/datasets/{dataset_id}/download")
def download_dataset(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset, _ = item_for(db, Dataset, dataset_id, current_user, "Dataset")
    return StreamingResponse(
        io.BytesIO(dataset.content),
        media_type=dataset.content_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(dataset.filename)}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset, _ = item_for(db, Dataset, dataset_id, current_user, "Dataset", EDITOR)
    db.query(AnalysisResult).filter(AnalysisResult.dataset_id == dataset.id).update({AnalysisResult.dataset_id: None})
    db.delete(dataset)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Saved analysis results
# ---------------------------------------------------------------------------
@router.post("/projects/{project_id}/results", status_code=status.HTTP_201_CREATED)
def save_result(project_id: int, body: ResultIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project, _ = project_for(db, project_id, current_user, EDITOR)
    if body.dataset_id is not None:
        dataset, _ = item_for(db, Dataset, body.dataset_id, current_user, "Dataset")
        if dataset.project_id != project.id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The dataset belongs to a different project.")
    size = len(json.dumps(body.result, separators=(",", ":")).encode("utf-8"))
    if body.report is not None:
        size += len(json.dumps(body.report, separators=(",", ":")).encode("utf-8"))
    if size > MAX_RESULT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Saved results can be at most {MAX_RESULT_BYTES // (1024 * 1024)} MB."
        )
    _require_quota(db, current_user, size)
    result = AnalysisResult(
        user_id=current_user.id, project_id=project.id, dataset_id=body.dataset_id, module=body.module,
        title=body.title, parameters=body.parameters, result=body.result, report=body.report, size_bytes=size,
    )
    db.add(result)
    project.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(result)
    return _result_json(result)


@router.get("/results/{result_id}")
def get_result(result_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result, _ = item_for(db, AnalysisResult, result_id, current_user, "Saved result")
    return _result_json(result, include_payload=True)


@router.delete("/results/{result_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_result(result_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result, _ = item_for(db, AnalysisResult, result_id, current_user, "Saved result", EDITOR)
    db.delete(result)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Data-preparation recipes
# ---------------------------------------------------------------------------
def _recipe_size(body: RecipeIn) -> int:
    size = len(json.dumps(body.steps, separators=(",", ":")).encode("utf-8")) + len(body.description or "")
    if size > MAX_RECIPE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Recipes can be at most {MAX_RECIPE_BYTES // 1024} KB."
        )
    return size


@router.get("/recipes")
def list_recipes(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Every recipe in the user's own and shared projects, newest first, with its project's name (for the Load-from-project list)."""
    accessible = accessible_project_ids(db, current_user)
    if not accessible:
        return []
    rows = (db.query(PrepRecipe, Project.name).join(Project, Project.id == PrepRecipe.project_id)
            .filter(PrepRecipe.project_id.in_(accessible)).order_by(PrepRecipe.updated_at.desc()).all())
    return [{**_recipe_json(recipe), "project_name": project_name} for recipe, project_name in rows]


@router.post("/projects/{project_id}/recipes", status_code=status.HTTP_201_CREATED)
def save_recipe(project_id: int, body: RecipeIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project, _ = project_for(db, project_id, current_user, EDITOR)
    size = _recipe_size(body)
    _require_quota(db, current_user, size)
    recipe = PrepRecipe(user_id=current_user.id, project_id=project.id, size_bytes=size, **body.model_dump())
    db.add(recipe)
    project.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(recipe)
    return _recipe_json(recipe)


@router.get("/recipes/{recipe_id}")
def get_recipe(recipe_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    recipe, _ = item_for(db, PrepRecipe, recipe_id, current_user, "Recipe")
    return _recipe_json(recipe)


@router.put("/recipes/{recipe_id}")
def replace_recipe(recipe_id: int, body: RecipeIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Save a changed recipe over the stored one (same project, same id)."""
    recipe, _ = item_for(db, PrepRecipe, recipe_id, current_user, "Recipe", EDITOR)
    size = _recipe_size(body)
    _require_quota(db, current_user, size - recipe.size_bytes)
    for field, value in body.model_dump().items():
        setattr(recipe, field, value)
    recipe.size_bytes = size
    recipe.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(recipe)
    return _recipe_json(recipe)


@router.delete("/recipes/{recipe_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recipe(recipe_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    recipe, _ = item_for(db, PrepRecipe, recipe_id, current_user, "Recipe", EDITOR)
    db.delete(recipe)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Project reports: methods text, references and one combined Word document
# ---------------------------------------------------------------------------
def _report_parts(db: Session, project: Project, result_ids, recipe_ids):
    datasets = db.query(Dataset).filter(Dataset.project_id == project.id).order_by(Dataset.created_at).all()
    results_query = db.query(AnalysisResult).filter(AnalysisResult.project_id == project.id)
    if result_ids is not None:
        results_query = results_query.filter(AnalysisResult.id.in_(result_ids))
    results = results_query.order_by(AnalysisResult.created_at).all()
    recipes_query = db.query(PrepRecipe).filter(PrepRecipe.project_id == project.id)
    if recipe_ids is not None:
        recipes_query = recipes_query.filter(PrepRecipe.id.in_(recipe_ids))
    recipes = recipes_query.order_by(PrepRecipe.created_at).all()
    return datasets, results, recipes


@router.get("/projects/{project_id}/methods")
def project_methods(project_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Methods paragraphs and references for every analysis and recipe in a project, ready to copy into a paper."""
    project, _ = project_for(db, project_id, current_user)
    _, results, recipes = _report_parts(db, project, None, None)
    keys, analyses, preparation = [], [], []
    for recipe in recipes:
        sentences, refs = methods_text.describe_recipe(recipe.steps)
        keys += refs
        preparation.append({"recipe_id": recipe.id, "name": recipe.name, "sentences": sentences})
    for result in results:
        described = methods_text.describe_analysis(result.module, result.parameters, result.result)
        keys += described["references"]
        analyses.append({"result_id": result.id, "title": result.title, "module": result.module, **described})
    software, refs = methods_text.software_paragraph()
    return {
        "preparation": preparation,
        "analyses": analyses,
        "software": software,
        "references": methods_text.reference_list(keys + refs),
    }


@router.post("/projects/{project_id}/report")
def project_report(project_id: int, body: ProjectReportIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project, _ = project_for(db, project_id, current_user)
    datasets, results, recipes = _report_parts(db, project, body.result_ids, body.recipe_ids)
    report = build_project_report(project, datasets, results, recipes, title=body.title,
                                  include_activity=body.include_activity, include_references=body.include_references)
    content = build_report(report, prepared_by=current_user.full_name or current_user.email)
    filename = re.sub(r"[^\w.-]+", "_", report.filename) + ".docx"
    return StreamingResponse(
        io.BytesIO(content),
        media_type=DOCX_MEDIA_TYPE,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )
