"""
Claude/Anthropic LLM provider implementation.
"""

from typing import Optional
from llm_log_analyzer.environment import PROVIDER_API_KEYS
from .base import LLMProvider, TEMPERATURE, MAX_OUTPUT_TOKENS
from .types import LLMProviderType

class ClaudeProvider(LLMProvider):
    """Claude/Anthropic LLM provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or PROVIDER_API_KEYS[LLMProviderType.CLAUDE.value], **kwargs)
        try:
            from anthropic import Anthropic
            if self.api_key:
                self.client = Anthropic(api_key=self.api_key)
            else:
                raise Exception("Claude API key is required")
        except ImportError:
            raise ImportError("anthropic package is required for Claude provider. Install requirements with: pip install -r requirements.txt")
    
    def get_default_chunk_model(self) -> str:
        return "claude-3-5-haiku-20241022"
    
    def get_default_aggregation_model(self) -> str:
        return "claude-3-5-sonnet-20241022"
    
    def get_default_tokenizer_encoding_name(self) -> str:
        return "cl100k_base"
    
    def _make_api_request(self, model: str, prompt: str) -> str:
        """Make request to Claude API."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            messages=messages
        )
        return response.content[0].text
