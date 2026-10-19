from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    PROJECT_NAME: str = "Stat Sathi"

    # Required. Set in backend/.env for local development (see .env.example)
    # or as environment variables on the hosting platform. Never hardcode them here.
    DATABASE_URL: str
    SECRET_KEY: str = Field(validation_alias="JWT_SECRET_KEY", min_length=32)

    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Where the web app lives, for the links in account emails (no trailing slash)
    FRONTEND_URL: str = "http://localhost:5173"

    # Account emails: "console" writes them to the server log (development); "smtp" sends them
    EMAIL_BACKEND: str = "console"
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM: Optional[str] = None          # e.g. "StatSathi <no-reply@example.org>"
    SMTP_SECURITY: str = "starttls"          # starttls (port 587), ssl (port 465) or none

    PASSWORD_RESET_MINUTES: int = 60
    EMAIL_VERIFICATION_HOURS: int = 48
    PASSWORD_MIN_LENGTH: int = 8

    # Limits on login, signup and email requests, kept in memory per server process
    RATE_LIMIT_ENABLED: bool = True
    # Behind a proxy (Render, a load balancer) the client address is in X-Forwarded-For
    TRUST_PROXY_HEADERS: bool = False

    # Browsers allowed to call the API: local development servers are always allowed; add the deployed
    # web app with CORS_ORIGINS (comma separated) and/or CORS_ORIGIN_REGEX. The default regex admits any
    # *.vercel.app site; set CORS_ORIGIN_REGEX to your own address (or empty) once the domain is fixed.
    CORS_ORIGINS: str = ""
    CORS_ORIGIN_REGEX: str = r"https://.*\.vercel\.app"

    # Largest request body accepted by any endpoint (the figure composer takes up to 45 MB of images)
    MAX_REQUEST_MB: int = 50
    # Largest CSV or Excel file an analysis will read
    MAX_TABLE_UPLOAD_MB: int = 25

    # Interactive API documentation at /docs and /redoc
    API_DOCS_ENABLED: bool = True

    @property
    def cors_origins(self):
        local = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"]
        extra = [origin.strip().rstrip("/") for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        return local + [origin for origin in extra if origin not in local]


settings = Settings()
