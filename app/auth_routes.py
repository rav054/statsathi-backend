import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import account_data
from . import account_security as security
from . import emailer
from .auth import create_access_token, find_user_by_email, get_current_user, get_password_hash, verify_password
from .config import settings
from .database import get_db
from .models import User
from .schemas import (ChangePasswordIn, EmailIn, MessageOut, ResetPasswordIn, Token, TokenIn, UserCreate, UserResponse,
                      UserUpdate)

logger = logging.getLogger("statsathi.auth")

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

RESET_REQUESTED = "If an account exists for that email address, a link to reset its password is on its way."


def _send_verification(db: Session, user: User):
    """Email a verification link; a failure to send is logged, never shown as a failed signup."""
    token = security.issue_token(db, user, security.PURPOSE_VERIFY, security.verification_lifetime())
    try:
        emailer.send_email_verification(user, token)
    except emailer.EmailError:
        logger.exception("Verification email for user %s was not sent", user.id)


def _session_after_password_change(db: Session, user: User):
    # Whole seconds: a session issued in the same second as the change must stay valid
    user.password_changed_at = datetime.utcnow().replace(microsecond=0)
    db.commit()
    return create_access_token({"sub": user.email}, timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user_in: UserCreate, request: Request, db: Session = Depends(get_db)):
    address_key = f"signup:{security.client_address(request)}"
    security.enforce(address_key, security.SIGNUPS_PER_ADDRESS, "Too many accounts were created from this network.")

    if find_user_by_email(db, user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    new_user = User(
        full_name=user_in.full_name,
        institution=user_in.institution,
        occupation=user_in.occupation,
        email=user_in.email.strip().lower(),
        hashed_password=get_password_hash(user_in.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    security.count(address_key, security.SIGNUPS_PER_ADDRESS)
    _send_verification(db, new_user)
    return new_user


@router.post("/login", response_model=Token)
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    email = (form_data.username or "").strip().lower()
    account_key = f"login-failures:{email}:{security.client_address(request)}"
    address_key = f"login:{security.client_address(request)}"
    security.enforce(account_key, security.LOGIN_FAILURES_PER_ACCOUNT,
                     "Too many failed sign-in attempts for this account.")
    security.enforce(address_key, security.LOGIN_ATTEMPTS_PER_ADDRESS, "Too many sign-in attempts from this network.")
    security.count(address_key, security.LOGIN_ATTEMPTS_PER_ADDRESS)

    user = find_user_by_email(db, email)
    if not user or not verify_password(form_data.password, user.hashed_password):
        security.count(account_key, security.LOGIN_FAILURES_PER_ACCOUNT)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    security.limiter.clear(account_key)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/forgot-password", response_model=MessageOut)
def forgot_password(body: EmailIn, request: Request, db: Session = Depends(get_db)):
    """Email a reset link. The answer is the same whether or not the address has an account."""
    email = body.email.strip().lower()
    email_key = f"reset:{email}"
    address_key = f"reset-address:{security.client_address(request)}"
    security.enforce(email_key, security.RESET_REQUESTS_PER_EMAIL, "Several reset links were already sent to this address.")
    security.enforce(address_key, security.RESET_REQUESTS_PER_ADDRESS, "Too many reset requests from this network.")
    security.count(email_key, security.RESET_REQUESTS_PER_EMAIL)
    security.count(address_key, security.RESET_REQUESTS_PER_ADDRESS)

    user = find_user_by_email(db, email)
    if user:
        token = security.issue_token(db, user, security.PURPOSE_RESET, security.reset_lifetime())
        try:
            emailer.send_password_reset(user, token)
        except emailer.EmailError:
            logger.exception("Password reset email for user %s was not sent", user.id)
    return {"message": RESET_REQUESTED}


@router.post("/reset-password", response_model=Token)
def reset_password(body: ResetPasswordIn, db: Session = Depends(get_db)):
    """Set a new password from a reset link. Every existing session ends; the reply signs the user in again."""
    record = security.consume_token(db, body.token, security.PURPOSE_RESET)
    if record is None:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="This reset link is invalid, has expired or has already been used. Ask for a new one.")
    user = db.get(User, record.user_id)
    user.hashed_password = get_password_hash(body.password)
    # Following the link proves the address belongs to the user
    if user.email_verified_at is None:
        user.email_verified_at = datetime.utcnow()
    token = _session_after_password_change(db, user)
    return {"access_token": token, "token_type": "bearer"}


@router.post("/verify-email", response_model=UserResponse)
def verify_email(body: TokenIn, db: Session = Depends(get_db)):
    record = security.consume_token(db, body.token, security.PURPOSE_VERIFY)
    if record is None:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="This verification link is invalid, has expired or has already been used.")
    user = db.get(User, record.user_id)
    if user.email_verified_at is None:
        user.email_verified_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user


@router.post("/resend-verification", response_model=MessageOut)
def resend_verification(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.email_verified_at is not None:
        return {"message": "Your email address is already verified."}
    key = f"verify:{current_user.id}"
    security.enforce(key, security.VERIFICATION_EMAILS_PER_USER, "Several verification emails were already sent.")
    security.count(key, security.VERIFICATION_EMAILS_PER_USER)
    _send_verification(db, current_user)
    return {"message": f"A verification link was sent to {current_user.email}."}


@router.post("/change-password", response_model=Token)
def change_password(body: ChangePasswordIn, request: Request, current_user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    key = f"change-password:{current_user.id}"
    security.enforce(key, security.LOGIN_FAILURES_PER_ACCOUNT, "Too many wrong current passwords.")
    if not verify_password(body.current_password, current_user.hashed_password):
        security.count(key, security.LOGIN_FAILURES_PER_ACCOUNT)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The current password is not correct.")
    current_user.hashed_password = get_password_hash(body.new_password)
    token = _session_after_password_change(db, current_user)
    return {"access_token": token, "token_type": "bearer"}


class DeleteAccountIn(BaseModel):
    password: str = Field(min_length=1, max_length=128)
    confirmation: str  # the user types DELETE


@router.get("/account/export")
def export_account(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Everything stored for the account as one ZIP file (see account_data for the layout)."""
    content = account_data.build_export(db, current_user)
    filename = f"statsathi-export-{datetime.utcnow():%Y-%m-%d}.zip"
    return Response(content=content, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/account/delete")
def delete_account(body: DeleteAccountIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Permanently delete the account. Needs the password and the word DELETE, so a stolen session alone cannot."""
    if body.confirmation.strip() != "DELETE":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Type DELETE to confirm.")
    key = f"delete-account:{current_user.id}"
    security.enforce(key, security.LOGIN_FAILURES_PER_ACCOUNT, "Too many wrong passwords.")
    if not verify_password(body.password, current_user.hashed_password):
        security.count(key, security.LOGIN_FAILURES_PER_ACCOUNT)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The password is not correct.")
    email, user_id = current_user.email, current_user.id
    summary = account_data.delete_account(db, current_user)
    logger.info("Account %s deleted: %s", user_id, summary)
    return {"message": f"The account {email} and its data have been deleted.", **summary}


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UserResponse)
def update_profile(profile_in: UserUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.full_name = profile_in.full_name
    current_user.institution = profile_in.institution
    current_user.occupation = profile_in.occupation
    db.commit()
    db.refresh(current_user)
    return current_user
