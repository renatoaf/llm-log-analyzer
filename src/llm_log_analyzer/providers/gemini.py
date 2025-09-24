"""
Google Gemini LLM provider implementation.
"""

from typing import Optional
from llm_log_analyzer.environment import PROVIDER_API_KEYS
from .base import LLMProvider, TEMPERATURE, MAX_OUTPUT_TOKENS
from .types import LLMProviderType

class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or PROVIDER_API_KEYS[LLMProviderType.GEMINI.value], **kwargs)
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                raise Exception("Gemini API key is required")
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package is required for Gemini provider. Install requirements with: pip install -r requirements.txt")

    
    def get_default_chunk_model(self) -> str:
        return "gemini-1.5-flash-latest"
    
    def get_default_aggregation_model(self) -> str:
        return "gemini-2.5-pro"
    
    def get_default_tokenizer_encoding_name(self) -> str:
        return "cl100k_base"

    def _make_api_request(self, model: str, prompt: str) -> str:
        """Make request to Gemini API."""
        model_instance = self.genai.GenerativeModel(model)
        
        generation_config = self.genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS
        )
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
        
        response = model_instance.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        if response.candidates and response.candidates[0].finish_reason == "SAFETY":
            raise Exception("Response was blocked by safety filters. Try rephrasing the input.")
        
        if not response.text:
            raise Exception("Empty response received from Gemini API")
        
        response_text = response.text.strip()
        self.logger.debug(f"Gemini response length: {len(response_text)} chars")
        
        return response_text
