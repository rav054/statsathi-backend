import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Stat Sathi"
    DATABASE_URL: str = "postgresql://postgres:Raviravi%405454@db.ranvkqyvjhvztywjexov.supabase.co:5432/postgres"
    
    # In production, these should be loaded from environment variables
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "b30fb0d33e5069a304ee0c6b1297e0169123fe387ad6ef2958e98031db26bcab")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days for development convenience

    class Config:
        env_file = ".env"

settings = Settings()
