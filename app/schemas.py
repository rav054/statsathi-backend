from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, computed_field

# The minimum is checked here for new passwords only; existing accounts keep signing in with the password they have
PASSWORD_MIN = 8
PASSWORD_MAX = 128


class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    institution: Optional[str] = None
    occupation: Optional[str] = None


class UserCreate(UserBase):
    full_name: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)


class UserUpdate(BaseModel):
    full_name: str
    institution: Optional[str] = None
    occupation: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: int
    created_at: datetime
    email_verified_at: Optional[datetime] = None

    @computed_field
    @property
    def email_verified(self) -> bool:
        return self.email_verified_at is not None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class MessageOut(BaseModel):
    message: str


class EmailIn(BaseModel):
    email: EmailStr


class TokenIn(BaseModel):
    token: str = Field(min_length=10, max_length=200)


class ResetPasswordIn(TokenIn):
    password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)


class ChangePasswordIn(BaseModel):
    current_password: str = Field(min_length=1, max_length=PASSWORD_MAX)
    new_password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)
