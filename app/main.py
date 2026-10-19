import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from .auth_routes import router as auth_router
from .analysis import router as analysis_router
from .sem_router import router as sem_router
from .workspace_routes import router as workspace_router
from .reports import router as reports_router
from .collaboration_routes import router as collaboration_router
from .config import settings
from .database import SessionLocal
from .hardening import RequestSizeLimitMiddleware, SecurityHeadersMiddleware, UnexpectedErrorMiddleware

# Initialize database tables
Base.metadata.create_all(bind=engine)

# Dynamically add occupation column if it doesn't exist (database migration fallback)
from sqlalchemy import inspect, text
inspector = inspect(engine)
if inspector.has_table("users"):
    columns = [col['name'] for col in inspector.get_columns('users')]
    if 'occupation' not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN occupation VARCHAR;"))
# Account security columns (added with password reset and email verification)
if inspector.has_table("users"):
    columns = [col['name'] for col in inspector.get_columns('users')]
    for column in ("email_verified_at", "password_changed_at"):
        if column not in columns:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {column} TIMESTAMP;"))
# Saved results keep their report content (added with project reports)
if inspector.has_table("analysis_results"):
    columns = [col['name'] for col in inspector.get_columns('analysis_results')]
    if 'report' not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE analysis_results ADD COLUMN report JSON;"))

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Your Trustworthy Research Analytics Companion.",
    version="1.0.0",
    docs_url="/docs" if settings.API_DOCS_ENABLED else None,
    redoc_url="/redoc" if settings.API_DOCS_ENABLED else None,
    openapi_url="/openapi.json" if settings.API_DOCS_ENABLED else None,
)

# Registered before CORS so that CORS wraps them: their 413 and 500 answers stay readable by the browser
app.add_middleware(UnexpectedErrorMiddleware)
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.MAX_REQUEST_MB * 1024 * 1024)
app.add_middleware(SecurityHeadersMiddleware)

# Configure CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Let the browser read how long to wait after a 429, and the file name of downloads
    expose_headers=["Retry-After", "Content-Disposition"],
)

# Register routers
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(sem_router)
app.include_router(workspace_router)
app.include_router(reports_router)
app.include_router(collaboration_router)

@app.get("/health")
def health():
    """For uptime checks: 200 when the database answers, 503 when it does not."""
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception:
        logging.getLogger("statsathi.errors").exception("Health check could not reach the database")
        return JSONResponse(status_code=503, content={"status": "unavailable", "database": "unreachable"})
    return {"status": "healthy", "database": "ok"}


@app.on_event("startup")
def warn_about_configuration():
    log = logging.getLogger("statsathi.config")
    local_frontend = "localhost" in settings.FRONTEND_URL or "127.0.0.1" in settings.FRONTEND_URL
    if settings.EMAIL_BACKEND == "console" and not local_frontend:
        log.warning("EMAIL_BACKEND is 'console': password reset and verification emails are only written to the log.")
    if settings.EMAIL_BACKEND == "smtp" and local_frontend:
        log.warning("FRONTEND_URL is %s: links in account emails will point to a local address.", settings.FRONTEND_URL)


@app.get("/")
def read_root():
    return {
        "app": settings.PROJECT_NAME,
        "status": "healthy",
        "description": "Backend API for statistical analysis and research visualization."
    }