"""
Configuration settings for Log Analyzer.
This file contains runtime configuration such as API keys.
"""

import os

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Provider-specific API keys and credentials
PROVIDER_API_KEYS = {
    "claude": ANTHROPIC_API_KEY,
    "gemini": GOOGLE_API_KEY,
    "openai": OPENAI_API_KEY,
    "aws-bedrock-claude": AWS_BEARER_TOKEN_BEDROCK
}

def get_provider_status() -> dict:
    """Get status of all LLM providers."""
    status = {}
    for provider, credentials in PROVIDER_API_KEYS.items():
        # All providers now use simple API key strings
        available = bool(credentials)
        preview = f"{credentials[:6]}..." if credentials else "Not set"
        status[provider] = {
            "available": available,
            "api_key_set": available,
            "api_key_preview": preview
        }
    return status
