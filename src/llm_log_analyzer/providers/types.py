"""
Provider types and enumerations.
"""

from enum import Enum

class LLMProviderType(Enum):
    """Supported LLM provider types."""
    CLAUDE = "claude"
    GEMINI = "gemini"
    OPENAI = "openai"
    AWS_BEDROCK_CLAUDE = "aws-bedrock-claude"
