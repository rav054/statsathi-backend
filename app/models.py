from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import deferred, relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    institution = Column(String, nullable=True)  # Institution/Affiliation
    occupation = Column(String, nullable=True)   # Occupation (Student, Researcher, etc.)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    email_verified_at = Column(DateTime, nullable=True)
    # Sessions issued before this moment are no longer accepted (set when the password changes)
    password_changed_at = Column(DateTime, nullable=True)


class AuthToken(Base):
    """A one-time link sent by email (password reset, email verification). Only a hash of the token is stored."""
    __tablename__ = "auth_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    purpose = Column(String(30), nullable=False)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)


class Project(Base):
    """A research project: task checklist, uploaded datasets and saved analysis results."""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    status = Column(String(30), nullable=False, default="Empty")
    samples = Column(Integer, nullable=False, default=0)
    variables = Column(Integer, nullable=False, default=0)
    test_used = Column(String(200), nullable=False, default="None")
    tasks = Column(JSON, nullable=False, default=list)  # [{"id", "text", "completed"}]
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    results = relationship("AnalysisResult", back_populates="project", cascade="all, delete-orphan")
    recipes = relationship("PrepRecipe", back_populates="project", cascade="all, delete-orphan")
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")


class ProjectMember(Base):
    """Someone the owner shared a project with. An invitation is addressed to an email and waits until that
    account (with the address confirmed) accepts it; only then is `user_id` set and the project opened to them."""
    __tablename__ = "project_members"
    __table_args__ = (UniqueConstraint("project_id", "email", name="uq_project_member_email"),)

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(320), nullable=False, index=True)  # lower-cased
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    role = Column(String(20), nullable=False, default="viewer")  # viewer | editor
    invited_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    accepted_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="members")


class Dataset(Base):
    """An uploaded CSV/Excel file stored with its project (the bytes are loaded only on download)."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(120), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    n_rows = Column(Integer, nullable=False)
    n_columns = Column(Integer, nullable=False)
    columns = Column(JSON, nullable=False, default=list)
    content = deferred(Column(LargeBinary, nullable=False))
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="datasets")


class AnalysisResult(Base):
    """A saved analysis: the module, its settings and the result payload returned by the API."""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    module = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    parameters = Column(JSON, nullable=False, default=dict)
    result = deferred(Column(JSON, nullable=False, default=dict))
    # The Word-report content the module built when the result was saved, reused in project reports
    report = deferred(Column(JSON, nullable=True))
    size_bytes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="results")


class PrepRecipe(Base):
    """A data-preparation recipe: the ordered cleaning steps, kept with the project so they can be run again."""
    __tablename__ = "prep_recipes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    source_dataset = Column(String(255), nullable=True)  # the file the recipe was written for, for reference
    steps = Column(JSON, nullable=False, default=list)
    size_bytes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="recipes")
