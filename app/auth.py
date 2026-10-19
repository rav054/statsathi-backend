import calendar
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import func
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from .models import User
from .schemas import TokenData

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Passwords are cut to 50 characters, as they always have been, so existing hashes keep matching
    return pwd_context.verify(plain_password[:50], hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password[:50])


def find_user_by_email(db: Session, email: str) -> Optional[User]:
    """Email addresses are compared without regard to case: Ravi@... and ravi@... are the same person."""
    return db.query(User).filter(func.lower(User.email) == (email or "").strip().lower()).first()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    now = datetime.utcnow()
    expire = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    # "iat" lets a password change end the sessions that were issued before it
    to_encode.update({"exp": expire, "iat": calendar.timegm(now.utctimetuple())})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    user = find_user_by_email(db, token_data.email)
    if user is None:
        raise credentials_exception
    if user.password_changed_at is not None:
        issued = payload.get("iat") or 0
        if issued < calendar.timegm(user.password_changed_at.utctimetuple()):
            raise credentials_exception  # signed in before the password was changed
    return user
