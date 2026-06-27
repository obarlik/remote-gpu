import hmac

from fastapi import Header, HTTPException

from gpu_server.config import AUTH_TOKEN


def require_token(authorization: str = Header(default="")) -> None:
    if not AUTH_TOKEN:
        raise HTTPException(500, "GPU_SERVER_TOKEN is not configured on the server")

    prefix = "Bearer "
    token = authorization[len(prefix):] if authorization.startswith(prefix) else ""
    if not token or not hmac.compare_digest(token, AUTH_TOKEN):
        raise HTTPException(401, "Invalid or missing bearer token")
