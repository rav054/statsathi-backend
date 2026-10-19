"""Account emails: password-reset and verification links.

With EMAIL_BACKEND=console (the default) nothing leaves the machine: the message is written to the server log, and kept
in `OUTBOX` so tests and local development can read the link. With EMAIL_BACKEND=smtp it is sent through the SMTP server
in the SMTP_* settings (for example a university mail server, Gmail with an app password, or a transactional email
service's SMTP relay).
"""

import logging
from html import escape
import smtplib
import ssl
from collections import deque
from email.message import EmailMessage
from email.utils import formataddr, parseaddr

from .config import settings

logger = logging.getLogger("statsathi.email")

OUTBOX = deque(maxlen=50)  # the most recent console emails


class EmailError(RuntimeError):
    """The email could not be sent."""


def _sender():
    configured = settings.SMTP_FROM or settings.SMTP_USERNAME or "no-reply@statsathi.local"
    name, address = parseaddr(configured)
    return formataddr((name or "StatSathi", address or configured))


def send_email(to, subject, text, html=None):
    message = EmailMessage()
    message["From"] = _sender()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(text)
    if html:
        message.add_alternative(html, subtype="html")

    backend = (settings.EMAIL_BACKEND or "console").lower()
    if backend == "console":
        OUTBOX.append({"to": to, "subject": subject, "text": text})
        logger.warning("Email (console backend, not sent) to %s: %s\n%s", to, subject, text)
        return
    if backend != "smtp":
        raise EmailError(f"Unknown EMAIL_BACKEND {settings.EMAIL_BACKEND!r}; use console or smtp.")
    if not settings.SMTP_HOST:
        raise EmailError("SMTP_HOST is not set.")

    security = (settings.SMTP_SECURITY or "starttls").lower()
    try:
        if security == "ssl":
            server = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, context=ssl.create_default_context(), timeout=20)
        else:
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20)
        with server:
            if security == "starttls":
                server.starttls(context=ssl.create_default_context())
            if settings.SMTP_USERNAME:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD or "")
            server.send_message(message)
    except (smtplib.SMTPException, OSError) as error:
        logger.error("Could not send email to %s: %s", to, error)
        raise EmailError(str(error))


def _html(heading, paragraphs, link, button, footer):
    body = "".join(f'<p style="margin:0 0 14px;line-height:1.5">{p}</p>' for p in paragraphs)
    return (
        '<div style="font-family:Arial,Helvetica,sans-serif;color:#1e293b;max-width:520px;margin:0 auto;padding:24px">'
        f'<h2 style="color:#4f46e5;margin:0 0 16px">{heading}</h2>{body}'
        f'<p style="margin:24px 0"><a href="{link}" style="background:#4f46e5;color:#ffffff;padding:12px 20px;'
        f'border-radius:10px;text-decoration:none;font-weight:bold">{button}</a></p>'
        f'<p style="font-size:12px;color:#64748b;line-height:1.5">If the button does not work, copy this address into '
        f'your browser:<br><a href="{link}" style="color:#4f46e5;word-break:break-all">{link}</a></p>'
        f'<p style="font-size:12px;color:#64748b;line-height:1.5">{footer}</p></div>'
    )


def send_password_reset(user, token):
    link = f"{settings.FRONTEND_URL.rstrip('/')}/reset-password?token={token}"
    minutes = settings.PASSWORD_RESET_MINUTES
    text = (
        f"Hello {user.full_name},\n\n"
        f"Someone asked to reset the password for your StatSathi account ({user.email}).\n"
        f"Choose a new password here (the link works once, for {minutes} minutes):\n\n{link}\n\n"
        "If you did not ask for this, ignore this email - your password stays as it is."
    )
    html = _html("Reset your StatSathi password",
                 [f"Hello {escape(user.full_name)},", f"Someone asked to reset the password for your account ({escape(user.email)})."],
                 link, "Choose a new password",
                 f"The link works once, for {minutes} minutes. If you did not ask for this, ignore this email.")
    send_email(user.email, "Reset your StatSathi password", text, html)


def send_project_invitation(to, inviter, project_name, role, has_account):
    link = f"{settings.FRONTEND_URL.rstrip('/')}/"
    what = ("open its datasets, saved results and reports" if role == "viewer"
            else "open it and add datasets, analyses and cleaning recipes")
    how = ("Sign in and accept the invitation on your Projects page." if has_account
           else f"Create a free StatSathi account with this email address ({to}), confirm the address, "
                "and the invitation will be waiting on your Projects page.")
    inviter_name = inviter.full_name or inviter.email
    subject = f"{inviter_name} shared the StatSathi project \"{project_name}\" with you"
    text = (
        f"Hello,\n\n{inviter_name} ({inviter.email}) invited you to the StatSathi project \"{project_name}\" as "
        f"{'a viewer' if role == 'viewer' else 'an editor'}, so you can {what}.\n\n{how}\n\n{link}\n\n"
        "If you do not know this person, ignore this email: nothing is shared until you accept."
    )
    html = _html("A project was shared with you",
                 [f"{escape(inviter_name)} ({escape(inviter.email)}) invited you to the project "
                  f"<strong>{escape(project_name)}</strong> as {'a viewer' if role == 'viewer' else 'an editor'}, "
                  f"so you can {what}.", escape(how)],
                 link, "Open StatSathi",
                 "If you do not know this person, ignore this email: nothing is shared until you accept.")
    send_email(to, subject, text, html)


def send_email_verification(user, token):
    link = f"{settings.FRONTEND_URL.rstrip('/')}/verify-email?token={token}"
    hours = settings.EMAIL_VERIFICATION_HOURS
    text = (
        f"Hello {user.full_name},\n\n"
        f"Please confirm that {user.email} is your email address, so you can recover your StatSathi account:\n\n"
        f"{link}\n\nThe link works for {hours} hours."
    )
    html = _html("Confirm your email address",
                 [f"Hello {escape(user.full_name)},",
                  f"Please confirm that {escape(user.email)} is your address, so you can always recover your StatSathi account."],
                 link, "Confirm my email", f"The link works for {hours} hours.")
    send_email(user.email, "Confirm your StatSathi email address", text, html)
