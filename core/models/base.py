"""
Base model client interface for all AI models.
Provides a unified interface for different model providers.
"""

import asyncio
import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import aiohttp
from enum import Enum


class ModelType(Enum):
    """Enum for different model types."""
    OLLAMA = "ollama"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"


@dataclass
class ModelResponse:
    """Standardized response from any model."""
    content: str
    model_used: str
    tokens_used: int
    cost: float
    response_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseModelClient(ABC):
    """Abstract base class for all model clients."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model", "unknown")
        self.timeout = config.get("timeout", 60)
        self.max_tokens = config.get("max_tokens", 4096)
        self.input_cost_per_token = config.get("input_cost_per_token", 0.0)
        self.output_cost_per_token = config.get("output_cost_per_token", 0.0)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self):
        """Initialize the client session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters:
                - temperature: float (default: 0.6)
                - max_tokens: int (default: from config)
                - system_prompt: Optional[str] (system instruction)
                - use_agents_prompt: bool (default: True) - whether to load system prompt from AGENTS.md
                - stream: bool (default: False) - whether to stream the response
                - stream_callback: Optional[callable] - callback for streaming chunks
                
        Returns:
            ModelResponse object
        """
        # Convert single prompt to conversation history format
        conversation_history = [{"role": "user", "content": prompt}]
        return await self.generate_with_history(conversation_history, **kwargs)
    
    async def analyze_image(self, image_path: str, prompt: str = "Describe this image in detail.", **kwargs) -> ModelResponse:
        """
        Analyze an image using the model's vision capabilities.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to accompany the image
            **kwargs: Additional parameters:
                - temperature: float (default: 0.6)
                - max_tokens: int (default: from config)
                - system_prompt: Optional[str] (system instruction)
                - use_agents_prompt: bool (default: True)
                
        Returns:
            ModelResponse object with image analysis
        """
        import base64
        from pathlib import Path
        
        # Read and encode image
        image_file = Path(image_path)
        if not image_file.exists():
            return ModelResponse(
                content="",
                model_used=self.model_name,
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error_message=f"Image file not found: {image_path}"
            )
        
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine image MIME type from file extension
        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        
        # Create vision message in OpenAI-compatible format
        vision_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                }
            ]
        }
        
        # Use generate_with_history with the vision message
        return await self.generate_with_history([vision_message], **kwargs)
    
    async def generate_with_history(self, conversation_history: List[Dict[str, Any]], **kwargs) -> ModelResponse:
        """
        Generate a response from the model using conversation history.
        
        Args:
            conversation_history: List of message dictionaries with 'role' and 'content' keys.
                                 Content can be string or list for multimodal messages.
            **kwargs: Additional parameters:
                - temperature: float (default: 0.6)
                - max_tokens: int (default: from config)
                - system_prompt: Optional[str] (system instruction)
                - use_agents_prompt: bool (default: True) - whether to load system prompt from AGENTS.md
                - stream: bool (default: False) - whether to stream the response
                - stream_callback: Optional[callable] - callback for streaming chunks
                
        Returns:
            ModelResponse object
        """
        logger = logging.getLogger(__name__)
        start_time = datetime.now()
        
        # Extract parameters
        temperature = kwargs.get("temperature", 0.6)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        system_prompt = ""
        use_agents_prompt = kwargs.get("use_agents_prompt", True)
        stream = kwargs.get("stream", False)
        stream_callback = kwargs.get("stream_callback", None)
        
        # Load system prompt from AGENTS.md if not provided and use_agents_prompt is True
        if use_agents_prompt and not system_prompt:
            from .prompt_loader import get_system_prompt
            system_prompt = get_system_prompt()
        
        # Prepare messages for API
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Prepare payload for API
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Prepare headers with API key
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Ensure session is initialized
            if self.session is None:
                await self.initialize()
            
            # Make API request
            async with self.session.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                
                if stream:
                    # Handle streaming response
                    content = ""
                    async for line in response.content:
                        if line:
                            # Parse streaming line
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if "choices" in chunk and len(chunk["choices"]) > 0:
                                        delta = chunk["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            chunk_content = delta["content"]
                                            content += chunk_content
                                            # Call stream callback if provided
                                            if stream_callback:
                                                await stream_callback(chunk_content)
                                except json.JSONDecodeError:
                                    continue
                    
                    # Estimate token counts for streaming response
                    # Use the last user message as prompt for estimation
                    last_user_message = ""
                    for msg in reversed(conversation_history):
                        if msg.get("role") == "user":
                            last_user_message = msg.get("content", "")
                            break
                    
                    token_counts = self._estimate_tokens_from_response({}, last_user_message)
                    cost = self.calculate_cost(token_counts)
                    response_time = (datetime.now() - start_time).total_seconds()
                    total_tokens = token_counts.get("input_tokens", 0) + token_counts.get("output_tokens", 0)
                    
                    return ModelResponse(
                        content=content,
                        model_used=self.model_name,
                        tokens_used=total_tokens,
                        cost=cost,
                        response_time=response_time,
                        success=True,
                        metadata={
                            "temperature": temperature,
                            "stream": True,
                            "token_details": token_counts
                        }
                    )
                else:
                    # Handle non-streaming response
                    result = await response.json()
                    
                    # Parse response
                    content = self._parse_openai_response(result)
                    
                    # Estimate tokens from conversation history and response
                    # Use the last user message as prompt for estimation
                    last_user_message = ""
                    for msg in reversed(conversation_history):
                        if msg.get("role") == "user":
                            last_user_message = msg.get("content", "")
                            break
                    
                    token_counts = self._estimate_tokens_from_response(result, last_user_message)
                    cost = self.calculate_cost(token_counts)
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    # Calculate total tokens for ModelResponse
                    total_tokens = token_counts.get("input_tokens", 0) + token_counts.get("output_tokens", 0)
                    
                    return ModelResponse(
                        content=content,
                        model_used=self.model_name,
                        tokens_used=total_tokens,
                        cost=cost,
                        response_time=response_time,
                        success=True,
                        metadata={
                            "temperature": temperature,
                            "stream": False,
                            "token_details": token_counts
                        }
                    )
            
        except Exception as e:
            logger.error(f"{self.model_type.value} generation failed: {str(e)}")
            response_time = (datetime.now() - start_time).total_seconds()
            
            return ModelResponse(
                content="",
                model_used=self.model_name,
                tokens_used=0,
                cost=0.0,
                response_time=response_time,
                success=False,
                error_message=str(e),
                metadata={"error_type": type(e).__name__}
            )
    
    @abstractmethod
    def calculate_cost(self, tokens_used: Dict[str, int]) -> float:
        """
        Calculate the cost for the given token counts.
        
        Args:
            tokens_used: Dictionary with 'input_tokens' and 'output_tokens' keys
            
        Returns:
            Cost in USD
        """
        pass
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate the response from the model API.
        
        Args:
            response: The API response
            
        Returns:
            True if response is valid
        """
        return response is not None and "choices" in response
    
    def extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract the content from the API response.
        
        Args:
            response: The API response
            
        Returns:
            Extracted content string
        """
        try:
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                elif "text" in choice:
                    return choice.get("text", "")
            return ""
        except (KeyError, IndexError, TypeError):
            return ""
    
    def extract_tokens(self, response: Dict[str, Any]) -> int:
        """
        Extract token usage from the API response.
        
        Args:
            response: The API response
            
        Returns:
            Number of tokens used
        """
        try:
            if "usage" in response:
                return response["usage"].get("total_tokens", 0)
            return 0
        except (KeyError, TypeError):
            return 0
    
    async def _make_request(self, url: str, headers: Dict[str, str], 
                           data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make an HTTP request to the model API.
        
        Args:
            url: API endpoint URL
            headers: HTTP headers
            data: Request data
            
        Returns:
            API response as dictionary
        """
        if self.session is None:
            await self.initialize()
        
        try:
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        "error": f"HTTP {response.status}: {error_text}",
                        "success": False
                    }
        except asyncio.TimeoutError:
            return {"error": "Request timeout", "success": False}
        except aiohttp.ClientError as e:
            return {"error": f"Client error: {str(e)}", "success": False}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}", "success": False}
    
    def _parse_openai_response(self, result: Dict[str, Any], response_format: Optional[Dict[str, Any]] = None) -> str:
        """
        Parse OpenAI-style API response.
        
        Args:
            result: API response
            response_format: Optional response format specification
            
        Returns:
            Extracted content string
        """
        try:
                
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    # If JSON format is requested, ensure it's valid JSON
                    if response_format and response_format.get("type") == "json_object":
                        try:
                            import json
                            # Try to parse as JSON to validate
                            res = json.loads(content)
                            # Get the first value from the JSON object
                            if isinstance(res, dict) and res:
                                return next(iter(res.values()))
                            else:
                                # If JSON is not a non-empty dict, return as-is
                                return content
                        except json.JSONDecodeError:
                            # If not valid JSON, wrap it
                            return json.dumps({"content": content})
                    
                    return content
            
            # Fallback: try legacy Ollama response format
            if "response" in result:
                return result["response"]
            elif "message" in result and "content" in result["message"]:
                return result["message"]["content"]
            
            # Final fallback: return raw response as string
            return str(result)
            
        except Exception:
            return str(result)
    
    def _estimate_tokens_from_response(self, result: Dict[str, Any], prompt: str = "") -> Dict[str, int]:
        """
        Estimate token usage from API response with separate input/output counts.
        
        Args:
            result: API response
            prompt: Original prompt (for fallback estimation)
            
        Returns:
            Dictionary with 'input_tokens' and 'output_tokens' keys
        """
        try:
            # Try to get token count from response
            if "usage" in result:
                usage = result["usage"]
                if "prompt_tokens" in usage and "completion_tokens" in usage:
                    return {
                        "input_tokens": usage["prompt_tokens"],
                        "output_tokens": usage["completion_tokens"]
                    }
                elif "total_tokens" in usage:
                    # If only total tokens available, estimate 70% input, 30% output
                    total = usage["total_tokens"]
                    input_tokens = int(total * 0.7)
                    output_tokens = total - input_tokens
                    return {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
            
            # Fallback: estimate from content
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    # Rough estimate: ~4 characters per token
                    output_tokens = max(1, len(content) // 4)
                    # Estimate input tokens from prompt
                    input_tokens = max(1, len(prompt) // 4)
                    return {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
            
            # Final fallback: estimate from prompt + response length
            response_text = str(result)
            total_chars = len(prompt) + len(response_text)
            total_tokens = max(1, total_chars // 4)
            # Split 50/50 as rough estimate
            input_tokens = total_tokens // 2
            output_tokens = total_tokens - input_tokens
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
            
        except Exception:
            # Default estimate
            return {
                "input_tokens": 50,
                "output_tokens": 50
            }
    
    def _extract_content_from_choice(self, choice: Dict[str, Any], response_format: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract content from a choice object with optional JSON processing.
        
        Args:
            choice: Choice object from API response
            response_format: Optional response format specification
            
        Returns:
            Extracted content string
        """
        if "message" in choice and "content" in choice["message"]:
            content = choice["message"]["content"]
            
            # If JSON format is requested, extract first value
            if response_format and response_format.get("type") == "json_object":
                return self._extract_first_json_value(content)
            
            return content
        
        return ""
    
    def _extract_first_json_value(self, content: str) -> str:
        """
        Extract the first value from a JSON string.
        
        Args:
            content: JSON string to parse
            
        Returns:
            First value from JSON, or original content if parsing fails
        """
        try:
            import json
            res = json.loads(content)
            # Get the first value from the JSON object
            if isinstance(res, dict) and res:
                return next(iter(res.values()))
            return content
        except json.JSONDecodeError:
            # If not valid JSON, wrap it in standard format
            return json.dumps({"content": content})
    
    def _extract_legacy_content(self, result: Dict[str, Any]) -> str:
        """
        Extract content from legacy response formats.
        
        Args:
            result: API response
            
        Returns:
            Extracted content or empty string
        """
        if "response" in result:
            return result["response"]
        elif "message" in result and "content" in result["message"]:
            return result["message"]["content"]
        return ""


class ModelClientFactory:
    """Factory for creating model clients."""
    
    @staticmethod
    def create_client(model_type: ModelType, config: Dict[str, Any]) -> BaseModelClient:
        """
        Create a model client based on the model type.
        
        Args:
            model_type: Type of model
            config: Model configuration
            
        Returns:
            Appropriate model client instance
        """
        if model_type == ModelType.OLLAMA:
            from core.models.ollama.client import OllamaClient
            return OllamaClient(config)
        elif model_type == ModelType.MISTRAL:
            from core.models.mistral.client import MistralClient
            return MistralClient(config)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Example configuration
    example_config = {
        "model": "test-model",
        "timeout": 30,
        "max_tokens": 4096,
        "output_cost_per_token": 0.000001
    }
    
    async def test_base_client():
        """Test the base client functionality."""
        client = BaseModelClient(example_config)
        print(f"Created client for model: {client.model_name}")
        print(f"Timeout: {client.timeout}s")
        print(f"Max tokens: {client.max_tokens}")
        print(f"Cost per token: ${client.output_cost_per_token}")
        
        # Test cost calculation
        token_counts = {"input_tokens": 700, "output_tokens": 300}
        cost = client.calculate_cost(token_counts)
        print(f"Cost for {token_counts['input_tokens']} input + {token_counts['output_tokens']} output tokens: ${cost}")
    
    asyncio.run(test_base_client())