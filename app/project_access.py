"""Who may do what in a project: the owner, editors and viewers.

- viewer: open the project, its saved results and recipes, download datasets, build methods text and reports
- editor: everything a viewer can, plus add datasets, results and recipes, update the task checklist and
  status, and change or delete what they added themselves
- owner: everything, including renaming or deleting the project, deleting anyone's items and managing members

Anyone without access gets 404, exactly as if the project did not exist, so project ids reveal nothing.
"""
from fastapi import HTTPException, status

from .models import Project, ProjectMember, User

VIEWER, EDITOR, OWNER = "viewer", "editor", "owner"
MEMBER_ROLES = (VIEWER, EDITOR)
_RANK = {VIEWER: 1, EDITOR: 2, OWNER: 3}

_REFUSALS = {
    EDITOR: "You can view this project but not change it. Ask the owner to make you an editor.",
    OWNER: "Only the project owner can do this.",
}


def role_in(db, project: Project, user: User):
    """The user's role in the project, or None."""
    if project.user_id == user.id:
        return OWNER
    member = (db.query(ProjectMember)
              .filter(ProjectMember.project_id == project.id, ProjectMember.user_id == user.id,
                      ProjectMember.accepted_at.isnot(None))
              .first())
    return member.role if member else None


def project_for(db, project_id: int, user: User, need: str = VIEWER):
    """(project, role) when the user has at least the `need` role; 404 without access, 403 with too little."""
    project = db.query(Project).filter(Project.id == project_id).first()
    role = role_in(db, project, user) if project is not None else None
    if role is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    if _RANK[role] < _RANK[need]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_REFUSALS[need])
    return project, role


def item_for(db, model, item_id: int, user: User, label: str, need: str = VIEWER):
    """(item, role) for a dataset, result or recipe, checked through the project it belongs to.

    With need=EDITOR an editor may only touch items they added; the owner may touch any item."""
    item = db.query(model).filter(model.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{label} not found.")
    try:
        _, role = project_for(db, item.project_id, user, need)
    except HTTPException as error:
        if error.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{label} not found.")
        raise
    if need == EDITOR and role == EDITOR and item.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Editors can change only what they added. Ask the project owner to change this {label.lower()}.")
    return item, role


def accessible_project_ids(db, user: User):
    """Ids of every project the user owns or has accepted an invitation to."""
    owned = [pid for (pid,) in db.query(Project.id).filter(Project.user_id == user.id).all()]
    shared = [pid for (pid,) in db.query(ProjectMember.project_id)
              .filter(ProjectMember.user_id == user.id, ProjectMember.accepted_at.isnot(None)).all()]
    return owned + [pid for pid in shared if pid not in owned]


def can_change(role: str, item, user: User) -> bool:
    return role == OWNER or (role == EDITOR and item.user_id == user.id)
