"""
Factory function for creating LLM provider instances.
"""

from typing import Optional, Union
from .types import LLMProviderType
from .gemini import GeminiProvider
from .claude import ClaudeProvider
from .openai import OpenAIProvider
from .bedrock import BedrockClaudeProvider

def create_llm_provider(provider: Union[LLMProviderType, str] = LLMProviderType.GEMINI, api_key: Optional[str] = None, **kwargs):
    """
    Initialize the LLM client with a specific provider.
    
    Args:
        provider: LLM provider to use (CLAUDE, GEMINI, OPENAI, AWS_BEDROCK_CLAUDE, or string)
        api_key: API key for the provider (AWS Bearer Token for Bedrock)
        **kwargs: Additional arguments passed to the provider
    """
    if isinstance(provider, str):
        provider = LLMProviderType(provider.lower())
    
    if provider == LLMProviderType.CLAUDE:
        return ClaudeProvider(api_key=api_key, **kwargs)
    elif provider == LLMProviderType.GEMINI:
        return GeminiProvider(api_key=api_key, **kwargs)
    elif provider == LLMProviderType.OPENAI:
        return OpenAIProvider(api_key=api_key, **kwargs)
    elif provider == LLMProviderType.AWS_BEDROCK_CLAUDE:
        return BedrockClaudeProvider(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
