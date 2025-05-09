import os
from fastapi import HTTPException, status
from fastapi.security import APIKeyHeader

# Define API key header name
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# Get API key from environment variable with a fallback for development
API_KEY = os.getenv("API_KEY", "your-secret-api-key-12345")

def validate_api_key(api_key: str):
    """Validate the API key."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": API_KEY_NAME},
        )
    return True