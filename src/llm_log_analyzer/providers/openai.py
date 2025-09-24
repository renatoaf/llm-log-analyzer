"""
OpenAI LLM provider implementation.
"""

from typing import Optional
from llm_log_analyzer.environment import PROVIDER_API_KEYS
from .base import LLMProvider, MAX_OUTPUT_TOKENS, TEMPERATURE
from .types import LLMProviderType

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or PROVIDER_API_KEYS[LLMProviderType.OPENAI.value], **kwargs)
        try:
            from openai import OpenAI
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
            else:
                raise Exception("OpenAI API key is required")
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider. Install requirements with: pip install -r requirements.txt")
    
    def get_default_chunk_model(self) -> str:
        return "gpt-4o-mini"
    
    def get_default_aggregation_model(self) -> str:
        return "gpt-4o-mini"
    
    def get_default_tokenizer_encoding_name(self) -> str:
        return "cl100k_base"
    
    def _make_api_request(self, model: str, prompt: str) -> str:
        """Make request to OpenAI Responses API."""
        response = self.client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE
        )
        
        if not response.output or not response.output[0].content:
            raise RuntimeError("Empty response received from OpenAI API")
        
        text_blocks = [
            block.text
            for item in response.output
            for block in item.content
            if block.type == "output_text"
        ]
        
        response_text = "".join(text_blocks).strip()

        self.logger.debug(f"OpenAI response length: {len(response_text)} chars")
        return response_text
