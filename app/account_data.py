"""A user's data as one ZIP file, and deleting an account with everything that belongs only to it.

Export layout (all JSON is UTF-8, dates ISO 8601):
  README.txt
  account.json                                  profile and sharing memberships
  projects/<id>-<name>/project.json             checklist, metadata and (for own projects) members
  projects/<id>-<name>/datasets/<id>-<file>     each stored dataset, byte for byte as uploaded
  projects/<id>-<name>/results/<id>-<title>.json   settings, full result and report content
  projects/<id>-<name>/recipes/<id>-<name>.json    in the recipe-file format Data Preparation loads
Own projects are exported whole; in projects shared with the user, only what the user added.
"""
import io
import json
import re
import zipfile
from datetime import datetime

from .models import AnalysisResult, AuthToken, Dataset, PrepRecipe, Project, ProjectMember, User

README = """StatSathi data export
=====================

Exported {when} UTC for {email}.

account.json lists your profile and the projects shared with you.
Each folder under projects/ is one project:
  project.json   the task checklist, metadata and who it is shared with
  datasets/      your uploaded files, exactly as uploaded
  results/       saved analyses: the settings, the full result and the report content
  recipes/       cleaning recipes; open them in Data Preparation with "Load recipe file"

Projects you own are complete. From projects someone else shared with you, only what you added is included.
"""


def _iso(value):
    return value.isoformat() if isinstance(value, datetime) else value


def _safe(text, limit=60):
    cleaned = re.sub(r"[^\w.\- ]+", "_", str(text or "")).strip(" ._")
    return (cleaned or "untitled")[:limit]


def _dump(data):
    return json.dumps(data, indent=2, ensure_ascii=False, default=_iso)


def build_export(db, user: User) -> bytes:
    owned = db.query(Project).filter(Project.user_id == user.id).order_by(Project.created_at).all()
    memberships = (db.query(ProjectMember, Project).join(Project, Project.id == ProjectMember.project_id)
                   .filter(ProjectMember.user_id == user.id, ProjectMember.accepted_at.isnot(None)).all())
    shared = [project for _, project in memberships]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", README.format(when=datetime.utcnow().strftime("%Y-%m-%d %H:%M"), email=user.email))
        archive.writestr("account.json", _dump({
            "email": user.email,
            "full_name": user.full_name,
            "institution": user.institution,
            "occupation": user.occupation,
            "created_at": user.created_at,
            "email_verified_at": user.email_verified_at,
            "projects_owned": [{"id": p.id, "name": p.name} for p in owned],
            "projects_shared_with_me": [
                {"id": project.id, "name": project.name, "role": member.role, "accepted_at": member.accepted_at}
                for member, project in memberships],
        }))

        for project in owned + shared:
            own = project.user_id == user.id
            folder = f"projects/{project.id}-{_safe(project.name)}"
            info = {
                "id": project.id, "name": project.name, "status": project.status, "samples": project.samples,
                "variables": project.variables, "test_used": project.test_used, "tasks": project.tasks or [],
                "created_at": project.created_at, "updated_at": project.updated_at,
                "yours": own,
            }
            if own:
                info["shared_with"] = [{"email": m.email, "role": m.role, "accepted": m.accepted_at is not None}
                                       for m in project.members]
            archive.writestr(f"{folder}/project.json", _dump(info))

            def mine(query, model):
                query = query.filter(model.project_id == project.id)
                return query.all() if own else query.filter(model.user_id == user.id).all()

            for dataset in mine(db.query(Dataset), Dataset):
                archive.writestr(f"{folder}/datasets/{dataset.id}-{_safe(dataset.filename, 120)}", dataset.content)
            for result in mine(db.query(AnalysisResult), AnalysisResult):
                archive.writestr(f"{folder}/results/{result.id}-{_safe(result.title)}.json", _dump({
                    "module": result.module, "title": result.title, "parameters": result.parameters,
                    "dataset_id": result.dataset_id, "created_at": result.created_at,
                    "result": result.result, "report": result.report,
                }))
            for recipe in mine(db.query(PrepRecipe), PrepRecipe):
                archive.writestr(f"{folder}/recipes/{recipe.id}-{_safe(recipe.name)}.json", _dump({
                    "statsathi_recipe": 1, "name": recipe.name, "description": recipe.description,
                    "dataset": recipe.source_dataset, "steps": recipe.steps,
                }))
    return buffer.getvalue()


def delete_account(db, user: User):
    """Remove the account, its own projects (with their datasets, results, recipes and members), its sign-in
    tokens and its memberships. What the user added to other people's projects stays there with the project owner,
    exactly as when a member leaves a project. Returns a summary of what was removed."""
    owned = db.query(Project).filter(Project.user_id == user.id).all()
    summary = {"projects": len(owned), "reassigned_items": 0}

    for model in (Dataset, AnalysisResult, PrepRecipe):
        items = db.query(model).filter(model.user_id == user.id, model.project_id.notin_([p.id for p in owned] or [-1])).all()
        for item in items:
            item.user_id = db.get(Project, item.project_id).user_id
        summary["reassigned_items"] += len(items)

    for project in owned:
        db.delete(project)  # cascades to its datasets, results, recipes and members
    db.flush()

    db.query(ProjectMember).filter(
        (ProjectMember.user_id == user.id) | (ProjectMember.email == user.email.lower())
    ).delete(synchronize_session=False)
    db.query(ProjectMember).filter(ProjectMember.invited_by == user.id).update(
        {ProjectMember.invited_by: None}, synchronize_session=False)
    db.query(AuthToken).filter(AuthToken.user_id == user.id).delete(synchronize_session=False)
    db.delete(user)
    db.commit()
    return summary
