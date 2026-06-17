from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from .auth_routes import router as auth_router
from .analysis import router as analysis_router
from .config import settings

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

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Your Trustworthy Research Analytics Companion.",
    version="1.0.0"
)

# Configure CORS to allow frontend communication
# In development, we can allow localhost origins
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://statsathi-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(analysis_router)

@app.get("/")
def read_root():
    return {
        "app": settings.PROJECT_NAME,
        "status": "healthy",
        "description": "Backend API for statistical analysis and research visualization."
    }
