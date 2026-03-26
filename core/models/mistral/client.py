"""
Mistral API client for high-precision model inference.
Specialized for structured output and complex reasoning tasks.
"""

import logging
from typing import Dict, Any, Optional

from ..base import BaseModelClient, ModelResponse, ModelType


logger = logging.getLogger(__name__)


class MistralClient(BaseModelClient):
    """Client for Mistral API for high-precision tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Mistral client.
        
        Args:
            config: Configuration dictionary from BotConfig.model_configs["mistral"]
        """
        super().__init__(config)
        self.model_type = ModelType.MISTRAL
        self.api_key = config.get("api_key", "")
        self.api_url = config.get("api_url", "https://api.mistral.ai")

    def calculate_cost(self, tokens_used: Dict[str, int]) -> float:
        """
        Calculate cost for Mistral API usage based on input and output tokens.
        
        Args:
            tokens_used: Dictionary with 'input_tokens' and 'output_tokens' keys
            
        Returns:
            Cost in USD
        """
        input_tokens = tokens_used.get("input_tokens", 0)
        output_tokens = tokens_used.get("output_tokens", 0)
        return (input_tokens * self.input_cost_per_token) + (output_tokens * self.output_cost_per_token)
    

    async def test_connection(self) -> bool:
        """
        Test connection to Mistral API.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.api_key:
            return False
            
        try:
            # Ensure session is initialized
            if self.session is None:
                await self.initialize()
            
            # Simple test request to check API key validity
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.get(
                f"{self.api_url}/v1/models",
                headers=headers,
                timeout=10
            ) as response:
                return response.status == 200
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Mistral model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "api_url": self.config["api_url"],
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "cost_per_token": self.cost_per_token,
            "enabled": self.config.get("enabled", True),
            "best_for": self.config.get("best_for", [])
        }


# Factory function for creating Mistral client
def create_mistral_client(config: Dict[str, Any]) -> MistralClient:
    """
    Create a Mistral client instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MistralClient instance
    """
    return MistralClient(config)