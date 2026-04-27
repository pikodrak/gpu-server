"""
API key authentication for the GPU server.
"""
import hmac

from fastapi import Header, HTTPException, status

from config import settings


async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> str:
    """
    Validates the API key from either the X-API-Key header or
    Authorization: Bearer <key> header (for OpenAI-compatible clients).
    Set GPU_SERVER_API_KEYS in .env as a comma-separated list.
    """
    key: str | None = x_api_key
    if key is None and authorization and authorization.startswith("Bearer "):
        key = authorization[len("Bearer "):].strip()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required: provide X-API-Key header or Authorization: Bearer <key>",
        )
    if not settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server has no API keys configured",
        )
    if not any(hmac.compare_digest(key, configured) for configured in settings.api_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return key
