"""Accounts: email verification, password reset and change, session invalidation, and rate limits.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import re
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

from tests.support import client_and_headers  # must come before other app imports

from app import account_security as security  # noqa: E402
from app import database, emailer  # noqa: E402
from app.config import settings  # noqa: E402
from app.models import AuthToken, User  # noqa: E402

PASSWORD = "field-trial-2026"


def token_from_last_email(kind):
    message = emailer.OUTBOX[-1]
    match = re.search(rf"/{kind}\?token=([A-Za-z0-9_\-]+)", message["text"])
    assert match, message["text"]
    return match.group(1)


class AccountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client, _ = client_and_headers()

    def setUp(self):
        emailer.OUTBOX.clear()
        security.limiter.clear()

    def signup(self, email, password=PASSWORD, name="Test Researcher"):
        return self.client.post("/auth/signup", json={"email": email, "password": password, "full_name": name})

    def login(self, email, password=PASSWORD):
        return self.client.post("/auth/login", data={"username": email, "password": password})

    def headers(self, token):
        return {"Authorization": f"Bearer {token}"}

    # -------------------------------------------------------------------------------------------- verification
    def test_signup_sends_a_verification_link_that_works_once(self):
        response = self.signup("verify-me@example.com")
        self.assertEqual(response.status_code, 201, response.text)
        self.assertFalse(response.json()["email_verified"])
        self.assertEqual(emailer.OUTBOX[-1]["to"], "verify-me@example.com")
        token = token_from_last_email("verify-email")

        verified = self.client.post("/auth/verify-email", json={"token": token})
        self.assertEqual(verified.status_code, 200, verified.text)
        self.assertTrue(verified.json()["email_verified"])
        self.assertEqual(self.client.post("/auth/verify-email", json={"token": token}).status_code, 400)

        session = self.login("verify-me@example.com").json()["access_token"]
        self.assertTrue(self.client.get("/auth/me", headers=self.headers(session)).json()["email_verified"])

    def test_resending_verification(self):
        self.signup("resend@example.com")
        first = token_from_last_email("verify-email")
        session = self.login("resend@example.com").json()["access_token"]
        response = self.client.post("/auth/resend-verification", headers=self.headers(session))
        self.assertIn("was sent to resend@example.com", response.json()["message"])
        second = token_from_last_email("verify-email")
        self.assertNotEqual(first, second)
        # a newer link replaces the older one
        self.assertEqual(self.client.post("/auth/verify-email", json={"token": first}).status_code, 400)
        self.assertEqual(self.client.post("/auth/verify-email", json={"token": second}).status_code, 200)
        again = self.client.post("/auth/resend-verification", headers=self.headers(session))
        self.assertIn("already verified", again.json()["message"])

    # ------------------------------------------------------------------------------------------ account rules
    def test_email_addresses_ignore_case(self):
        self.assertEqual(self.signup("Mixed.Case@Example.com").json()["email"], "mixed.case@example.com")
        self.assertEqual(self.login("MIXED.CASE@example.COM").status_code, 200)
        self.assertEqual(self.signup("mixed.case@EXAMPLE.com").status_code, 400)

    def test_new_passwords_need_eight_characters(self):
        self.assertEqual(self.signup("short@example.com", password="1234567").status_code, 422)
        self.assertEqual(self.signup("empty@example.com", password="").status_code, 422)

    def test_html_email_escapes_the_display_name(self):
        with mock.patch.object(emailer, "send_email") as send:
            user = User(full_name="<script>alert(1)</script>", email="x@example.com")
            emailer.send_password_reset(user, "token-value")
        html = send.call_args[0][3]
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    # ----------------------------------------------------------------------------------------- password reset
    def test_reset_request_looks_the_same_for_unknown_addresses(self):
        known = self.signup("known@example.com")
        self.assertEqual(known.status_code, 201)
        emailer.OUTBOX.clear()
        for email in ("known@example.com", "nobody@example.com"):
            response = self.client.post("/auth/forgot-password", json={"email": email})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["message"], "If an account exists for that email address, a link to reset its password is on its way.")
        self.assertEqual([m["to"] for m in emailer.OUTBOX], ["known@example.com"])

    def test_reset_sets_the_password_ends_old_sessions_and_signs_in(self):
        self.signup("reset-me@example.com")
        old_session = self.login("reset-me@example.com").json()["access_token"]
        self.assertEqual(self.client.get("/auth/me", headers=self.headers(old_session)).status_code, 200)

        self.client.post("/auth/forgot-password", json={"email": "Reset-Me@example.com"})
        token = token_from_last_email("reset-password")
        time.sleep(1.1)  # sessions are compared to the change in whole seconds
        response = self.client.post("/auth/reset-password", json={"token": token, "password": "new-password-2026"})
        self.assertEqual(response.status_code, 200, response.text)
        new_session = response.json()["access_token"]

        self.assertEqual(self.client.get("/auth/me", headers=self.headers(old_session)).status_code, 401)
        me = self.client.get("/auth/me", headers=self.headers(new_session))
        self.assertEqual(me.status_code, 200)
        self.assertTrue(me.json()["email_verified"])  # the reset link proved the address
        self.assertEqual(self.login("reset-me@example.com").status_code, 401)
        self.assertEqual(self.login("reset-me@example.com", "new-password-2026").status_code, 200)
        # the link works once
        self.assertEqual(self.client.post("/auth/reset-password", json={"token": token, "password": "another-password"}).status_code, 400)

    def test_expired_replaced_and_forged_reset_links_are_refused(self):
        self.signup("expiry@example.com")
        self.client.post("/auth/forgot-password", json={"email": "expiry@example.com"})
        first = token_from_last_email("reset-password")
        self.client.post("/auth/forgot-password", json={"email": "expiry@example.com"})
        second = token_from_last_email("reset-password")
        self.assertEqual(self.client.post("/auth/reset-password", json={"token": first, "password": "password-one"}).status_code, 400)

        db = database.SessionLocal()
        try:
            for record in db.query(AuthToken).filter(AuthToken.used_at.is_(None)).all():
                record.expires_at = datetime.utcnow() - timedelta(minutes=1)
            db.commit()
        finally:
            db.close()
        self.assertEqual(self.client.post("/auth/reset-password", json={"token": second, "password": "password-two"}).status_code, 400)
        self.assertEqual(self.client.post("/auth/reset-password", json={"token": "x" * 43, "password": "password-three"}).status_code, 400)
        self.assertEqual(self.client.post("/auth/reset-password", json={"token": second, "password": "short"}).status_code, 422)

    def test_tokens_are_stored_hashed(self):
        self.signup("hashed@example.com")
        token = token_from_last_email("verify-email")
        db = database.SessionLocal()
        try:
            stored = [record.token_hash for record in db.query(AuthToken).all()]
        finally:
            db.close()
        self.assertNotIn(token, stored)
        self.assertTrue(all(len(value) == 64 for value in stored))

    def test_an_email_failure_does_not_break_signup_or_reveal_accounts(self):
        with mock.patch.object(emailer, "send_email", side_effect=emailer.EmailError("SMTP down")):
            self.assertEqual(self.signup("smtp-down@example.com").status_code, 201)
            response = self.client.post("/auth/forgot-password", json={"email": "smtp-down@example.com"})
        self.assertEqual(response.status_code, 200)

    # ---------------------------------------------------------------------------------------- change password
    def test_change_password(self):
        self.signup("change@example.com")
        session = self.login("change@example.com").json()["access_token"]
        wrong = self.client.post("/auth/change-password", headers=self.headers(session),
                                 json={"current_password": "not-it", "new_password": "changed-2026"})
        self.assertEqual(wrong.status_code, 400)
        time.sleep(1.1)
        changed = self.client.post("/auth/change-password", headers=self.headers(session),
                                   json={"current_password": PASSWORD, "new_password": "changed-2026"})
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertEqual(self.client.get("/auth/me", headers=self.headers(session)).status_code, 401)
        self.assertEqual(self.client.get("/auth/me", headers=self.headers(changed.json()["access_token"])).status_code, 200)
        self.assertEqual(self.login("change@example.com", "changed-2026").status_code, 200)

    # ------------------------------------------------------------------------------------------- rate limits
    def test_failed_logins_lock_the_account_for_that_address_only(self):
        self.signup("locked@example.com")
        self.signup("other@example.com")
        with mock.patch.object(settings, "RATE_LIMIT_ENABLED", True):
            security.limiter.clear()
            for _ in range(5):
                self.assertEqual(self.login("locked@example.com", "wrong-password").status_code, 401)
            blocked = self.login("locked@example.com")  # even the right password waits
            self.assertEqual(blocked.status_code, 429)
            self.assertIn("Retry-After", blocked.headers)
            self.assertIn("try again", blocked.json()["detail"])
            self.assertEqual(self.login("other@example.com").status_code, 200)

    def test_a_successful_login_resets_the_failure_count(self):
        self.signup("forgetful@example.com")
        with mock.patch.object(settings, "RATE_LIMIT_ENABLED", True):
            security.limiter.clear()
            for _ in range(4):
                self.login("forgetful@example.com", "wrong-password")
            self.assertEqual(self.login("forgetful@example.com").status_code, 200)
            for _ in range(4):
                self.assertEqual(self.login("forgetful@example.com", "wrong-password").status_code, 401)
            self.assertEqual(self.login("forgetful@example.com").status_code, 200)

    def test_reset_requests_are_limited(self):
        with mock.patch.object(settings, "RATE_LIMIT_ENABLED", True):
            security.limiter.clear()
            codes = [self.client.post("/auth/forgot-password", json={"email": "flood@example.com"}).status_code for _ in range(4)]
        self.assertEqual(codes, [200, 200, 200, 429])

    def test_sliding_window_and_proxy_address(self):
        limiter = security.RateLimiter()
        # two events at t=0, checked at t=1 (still inside the 60-second window) and at t=61 (outside it)
        with mock.patch.object(security.time, "monotonic", side_effect=[0, 0, 1, 61]):
            limiter.record("k", 60)
            limiter.record("k", 60)
            self.assertGreater(limiter.blocked_for("k", 2, 60), 0)
            self.assertEqual(limiter.blocked_for("k", 2, 60), 0)  # the first event has left the window

        request = mock.Mock()
        request.headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
        request.client.host = "10.0.0.1"
        self.assertEqual(security.client_address(request), "10.0.0.1")
        with mock.patch.object(settings, "TRUST_PROXY_HEADERS", True):
            self.assertEqual(security.client_address(request), "203.0.113.9")


if __name__ == "__main__":
    unittest.main()
