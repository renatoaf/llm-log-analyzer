"""
AWS Bedrock LLM provider implementations.
"""

from abc import abstractmethod
from typing import Dict, Optional
from llm_log_analyzer.environment import PROVIDER_API_KEYS, AWS_REGION
from .base import LLMProvider, TEMPERATURE, MAX_OUTPUT_TOKENS
from .types import LLMProviderType

class BedrockProvider(LLMProvider):
    """Abstract base class for AWS Bedrock LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or PROVIDER_API_KEYS[LLMProviderType.AWS_BEDROCK_CLAUDE.value], **kwargs)
        try:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 package is required for Bedrock providers. Install requirements with: pip install -r requirements.txt")
            
            if not self.api_key:
                raise Exception("AWS Bearer Token for Bedrock is required")
            
            import os
            os.environ['AWS_BEARER_TOKEN_BEDROCK'] = self.api_key
            
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=AWS_REGION
            )
            
        except Exception as e:
            raise Exception(f"Failed to initialize AWS Bedrock client: {e}")
    
    def _invoke_bedrock_model(self, model_id: str, body: Dict) -> str:
        """Common method to invoke Bedrock models."""
        import json
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            return self._extract_response_text(response_body)
            
        except Exception as e:
            self.logger.error(f"Bedrock model invocation failed: {e}")
            raise
    
    @abstractmethod
    def _extract_response_text(self, response_body: Dict) -> str:
        """Extract text from the response body. Implemented by model-specific subclasses."""
        pass
    
    @abstractmethod
    def _format_request_body(self, prompt: str) -> Dict:
        """Format the request body for the specific model. Implemented by model-specific subclasses."""
        pass


class BedrockClaudeProvider(BedrockProvider):
    """AWS Bedrock Claude LLM provider implementation."""
    
    def get_default_chunk_model(self) -> str:
        return "anthropic.claude-3-5-haiku-20241022-v1:0"
    
    def get_default_aggregation_model(self) -> str:
        return "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    def get_default_tokenizer_encoding_name(self) -> str:
        return "cl100k_base"
    
    def _make_api_request(self, model: str, prompt: str) -> str:
        """Make request to Claude via Bedrock."""
        body = self._format_request_body(prompt)
        return self._invoke_bedrock_model(model, body)
    
    def _format_request_body(self, prompt: str) -> Dict:
        """Format request body for Claude models on Bedrock."""
        messages = [
            {"role": "user", "content": prompt}
        ]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "messages": messages
        }
        
        return body
    
    def _extract_response_text(self, response_body: Dict) -> str:
        """Extract text from Claude Bedrock response."""
        try:
            content = response_body.get('content', [])
            if content and isinstance(content, list):
                return content[0].get('text', '')
            return ''
        except (KeyError, IndexError, AttributeError) as e:
            raise Exception(f"Failed to extract text from Claude response: {e}")
