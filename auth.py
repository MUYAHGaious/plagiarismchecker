import os
from dotenv import load_dotenv
from fastapi import HTTPException, status
from fastapi.security import APIKeyHeader

# Load environment variables from .env file
load_dotenv()

# Define API key header name
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# Get API key from environment variable
API_KEY = os.getenv("API_KEY", "default-fallback-key")

def validate_api_key(api_key: str):
    """Validate the API key."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": API_KEY_NAME},
        )
    return True