from fastapi import Header, HTTPException
import os

# API Key - can be set via environment variable or default
API_KEY = os.getenv("API_KEY", "sk_multipass_987654321")


def verify_api_key(x_api_key: str = Header(None, alias="x-api-key")):
    """
    Verify the API key from x-api-key header.
    Competition format: x-api-key: YOUR_SECRET_API_KEY
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=401, 
            detail="Missing API key. Include 'x-api-key' header."
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key or malformed request"
        )
