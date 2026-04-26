"""
API key authentication for the GPU server.
"""
import hmac

from fastapi import Header, HTTPException, status

from config import settings


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Validates the X-API-Key header against the configured API keys.
    Set GPU_SERVER_API_KEYS in .env as a comma-separated list.
    """
    if not settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server has no API keys configured",
        )
    if not any(hmac.compare_digest(x_api_key, key) for key in settings.api_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key
