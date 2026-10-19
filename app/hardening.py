"""Request-level protection: body size limits, safe answers to unexpected errors and security headers.

These are plain ASGI middlewares so they also cover streamed request bodies and never buffer responses.
Register them inside CORS (add them before CORSMiddleware) so their error answers still reach the browser.
"""
import json
import logging
import secrets

logger = logging.getLogger("statsathi.errors")

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    (b"cross-origin-resource-policy", b"cross-origin"),
]


async def _send_json(send, status, body, extra_headers=()):
    payload = json.dumps(body).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode()),
                    *extra_headers],
    })
    await send({"type": "http.response.body", "body": payload})


class _BodyTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    """Refuse request bodies over `max_bytes` with 413, whether or not the client declared the length."""

    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    def _message(self):
        return {"detail": f"This upload is too large. Requests can be at most {self.max_bytes // (1024 * 1024)} MB."}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    return await _send_json(send, 400, {"detail": "Invalid Content-Length header."})
                if declared > self.max_bytes:
                    return await _send_json(send, 413, self._message())

        received = 0
        started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await _send_json(send, 413, self._message())


class UnexpectedErrorMiddleware:
    """Answer an unhandled exception with a JSON 500 that names a reference, and log the traceback under it.

    Without this the browser gets a bare 500 without CORS headers, which the web app can only report as
    a network failure, and nothing ties the user's report to the server log.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = False

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except _BodyTooLarge:
            raise  # answered with 413 by RequestSizeLimitMiddleware
        except Exception:
            reference = secrets.token_hex(4)
            logger.exception("Unexpected error %s on %s %s", reference, scope.get("method"), scope.get("path"))
            if started:
                raise
            await _send_json(send, 500, {
                "detail": "Something went wrong on the server while handling this request. Please try again; "
                          f"if it keeps happening, report reference {reference}.",
                "reference": reference,
            })


class SecurityHeadersMiddleware:
    """Add conservative security headers to every response (HSTS only when the request came over HTTPS)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        https = scope.get("scheme") == "https" or any(
            name == b"x-forwarded-proto" and value.split(b",")[0].strip() == b"https"
            for name, value in scope.get("headers", []))

        async def with_headers(message):
            if message["type"] == "http.response.start":
                present = {name.lower() for name, _ in message.get("headers", [])}
                headers = list(message.get("headers", []))
                headers += [(name, value) for name, value in SECURITY_HEADERS if name not in present]
                if https and b"strict-transport-security" not in present:
                    headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, with_headers)
