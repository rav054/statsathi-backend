"""One-time email tokens and request rate limits for the account endpoints."""

import calendar
import hashlib
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

from fastapi import HTTPException, Request, status

from .config import settings
from .models import AuthToken

PURPOSE_RESET = "password_reset"
PURPOSE_VERIFY = "email_verification"


def utc_timestamp(moment):
    """Seconds since the epoch for a naive UTC datetime (datetime.timestamp() would assume local time)."""
    return calendar.timegm(moment.utctimetuple())


def _hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_token(db, user, purpose, lifetime):
    """A new random token for the user; earlier unused tokens for the same purpose stop working."""
    now = datetime.utcnow()
    db.query(AuthToken).filter(AuthToken.user_id == user.id, AuthToken.purpose == purpose,
                               AuthToken.used_at.is_(None)).update({AuthToken.used_at: now})
    raw = secrets.token_urlsafe(32)
    db.add(AuthToken(user_id=user.id, purpose=purpose, token_hash=_hash(raw), created_at=now, expires_at=now + lifetime))
    db.commit()
    return raw


def consume_token(db, raw, purpose):
    """The token's record if it is valid for this purpose, marked as used; otherwise None."""
    if not raw or len(raw) > 200:
        return None
    record = db.query(AuthToken).filter(AuthToken.token_hash == _hash(raw), AuthToken.purpose == purpose).first()
    now = datetime.utcnow()
    if record is None or record.used_at is not None or record.expires_at <= now:
        return None
    record.used_at = now
    return record


def reset_lifetime():
    return timedelta(minutes=settings.PASSWORD_RESET_MINUTES)


def verification_lifetime():
    return timedelta(hours=settings.EMAIL_VERIFICATION_HOURS)


# ------------------------------------------------------------------------------------------------ rate limiting

class RateLimiter:
    """Sliding-window counters kept in memory. Each key allows `limit` events per `window` seconds."""

    def __init__(self):
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, events, window, now):
        while events and events[0] <= now - window:
            events.popleft()

    def blocked_for(self, key, limit, window):
        """Seconds until the next event is allowed (0 when it is allowed now), without recording anything."""
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            self._prune(events, window, now)
            if len(events) < limit:
                return 0
            return max(1, int(events[0] + window - now) + 1)

    def record(self, key, window):
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            self._prune(events, window, now)
            events.append(now)

    def clear(self, key=None):
        with self._lock:
            if key is None:
                self._events.clear()
            else:
                self._events.pop(key, None)


limiter = RateLimiter()

# (limit, window in seconds)
LOGIN_FAILURES_PER_ACCOUNT = (5, 15 * 60)
LOGIN_ATTEMPTS_PER_ADDRESS = (30, 15 * 60)
SIGNUPS_PER_ADDRESS = (10, 60 * 60)
RESET_REQUESTS_PER_EMAIL = (3, 60 * 60)
RESET_REQUESTS_PER_ADDRESS = (10, 60 * 60)
VERIFICATION_EMAILS_PER_USER = (3, 60 * 60)
INVITATIONS_PER_USER = (30, 60 * 60)


def client_address(request: Request):
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce(key, rule, message):
    """Raise 429 when the key has used up its allowance; nothing is recorded here."""
    if not settings.RATE_LIMIT_ENABLED:
        return
    wait = limiter.blocked_for(key, *rule)
    if wait:
        minutes = max(1, round(wait / 60))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"{message} Please try again in about {minutes} minute{'s' if minutes != 1 else ''}.",
            headers={"Retry-After": str(wait)},
        )


def count(key, rule):
    if settings.RATE_LIMIT_ENABLED:
        limiter.record(key, rule[1])
