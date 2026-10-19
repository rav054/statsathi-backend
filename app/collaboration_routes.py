"""Sharing projects: the owner invites people by email as viewers or editors; they accept from their Projects page."""
import logging
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import func
from sqlalchemy.orm import Session

from . import account_security, emailer
from .auth import get_current_user
from .database import get_db
from .models import AnalysisResult, Dataset, PrepRecipe, Project, ProjectMember, User
from .project_access import OWNER, project_for
from .workspace_routes import member_json

router = APIRouter(prefix="/workspace", tags=["Collaboration"])
logger = logging.getLogger("statsathi.collaboration")

MAX_MEMBERS_PER_PROJECT = 25


class InviteIn(BaseModel):
    email: EmailStr
    role: Literal["viewer", "editor"] = "viewer"


class RoleIn(BaseModel):
    role: Literal["viewer", "editor"]


def _member(db: Session, project_id: int, member_id: int):
    member = db.query(ProjectMember).filter(ProjectMember.id == member_id, ProjectMember.project_id == project_id).first()
    if member is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found.")
    return member


def _account_for(db: Session, email: str):
    return db.query(User).filter(func.lower(User.email) == email).first()


@router.post("/projects/{project_id}/members", status_code=status.HTTP_201_CREATED)
def invite_member(project_id: int, body: InviteIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project, _ = project_for(db, project_id, current_user, OWNER)
    email = body.email.strip().lower()
    if email == current_user.email.lower():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You already own this project.")
    existing = db.query(ProjectMember).filter(ProjectMember.project_id == project.id, ProjectMember.email == email).first()
    if existing is not None:
        state = "is already a member of" if existing.accepted_at else "has already been invited to"
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"{email} {state} this project.")
    if db.query(ProjectMember).filter(ProjectMember.project_id == project.id).count() >= MAX_MEMBERS_PER_PROJECT:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"A project can be shared with at most {MAX_MEMBERS_PER_PROJECT} people.")
    limit_key = f"invite:user:{current_user.id}"
    account_security.enforce(limit_key, account_security.INVITATIONS_PER_USER, "You have sent many invitations recently.")

    member = ProjectMember(project_id=project.id, email=email, role=body.role, invited_by=current_user.id)
    db.add(member)
    db.commit()
    db.refresh(member)
    account_security.count(limit_key, account_security.INVITATIONS_PER_USER)

    email_sent = True
    try:
        emailer.send_project_invitation(email, current_user, project.name, body.role, _account_for(db, email) is not None)
    except emailer.EmailError:
        email_sent = False
        logger.warning("Invitation to project %s saved but the email to %s could not be sent", project.id, email)
    return {**member_json(member), "email_sent": email_sent}


@router.patch("/projects/{project_id}/members/{member_id}")
def change_member_role(project_id: int, member_id: int, body: RoleIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    project, _ = project_for(db, project_id, current_user, OWNER)
    member = _member(db, project.id, member_id)
    member.role = body.role
    db.commit()
    db.refresh(member)
    return member_json(member, db.get(User, member.user_id) if member.user_id else None)


@router.delete("/projects/{project_id}/members/{member_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(project_id: int, member_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """The owner removes someone or cancels an invitation, or a member leaves the project.

    What the member added stays in the project and passes to the owner (and counts towards the owner's storage)."""
    project, role = project_for(db, project_id, current_user)
    member = _member(db, project.id, member_id)
    if role != OWNER and member.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the project owner can remove other people.")
    if member.user_id is not None:
        for model in (Dataset, AnalysisResult, PrepRecipe):
            (db.query(model).filter(model.project_id == project.id, model.user_id == member.user_id)
             .update({model.user_id: project.user_id}, synchronize_session=False))
    db.delete(member)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Invitations addressed to the signed-in user
# ---------------------------------------------------------------------------
def _pending_for(db: Session, user: User):
    return (db.query(ProjectMember)
            .filter(ProjectMember.email == user.email.lower(), ProjectMember.accepted_at.is_(None))
            .order_by(ProjectMember.created_at.desc()).all())


@router.get("/invitations")
def my_invitations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    pending = _pending_for(db, current_user)
    if current_user.email_verified_at is None:
        # Anyone can sign up with any address; details wait until the address is confirmed as theirs
        return {"invitations": [], "pending_count": len(pending), "needs_verification": bool(pending)}
    projects = {p.id: p for p in db.query(Project).filter(Project.id.in_([m.project_id for m in pending])).all()} if pending else {}
    inviter_ids = {m.invited_by for m in pending} | {p.user_id for p in projects.values()}
    people = {u.id: u for u in db.query(User).filter(User.id.in_(inviter_ids)).all()} if inviter_ids else {}
    invitations = []
    for m in pending:
        project = projects.get(m.project_id)
        if project is None:
            continue
        inviter = people.get(m.invited_by) or people.get(project.user_id)
        invitations.append({
            "id": m.id,
            "project_id": project.id,
            "project_name": project.name,
            "role": m.role,
            "invited_by": {"full_name": inviter.full_name, "email": inviter.email} if inviter else None,
            "invited_at": m.created_at.isoformat() if m.created_at else None,
        })
    return {"invitations": invitations, "pending_count": len(invitations), "needs_verification": False}


def _my_invitation(db: Session, invitation_id: int, user: User):
    member = db.query(ProjectMember).filter(ProjectMember.id == invitation_id).first()
    if member is None or member.email != user.email.lower() or member.accepted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found.")
    return member


@router.post("/invitations/{invitation_id}/accept")
def accept_invitation(invitation_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    member = _my_invitation(db, invitation_id, current_user)
    if current_user.email_verified_at is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Confirm your email address first, using the link we sent you, then accept the invitation.")
    member.user_id = current_user.id
    member.accepted_at = datetime.utcnow()
    db.commit()
    return {"project_id": member.project_id, "role": member.role}


@router.post("/invitations/{invitation_id}/decline", status_code=status.HTTP_204_NO_CONTENT)
def decline_invitation(invitation_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(_my_invitation(db, invitation_id, current_user))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
