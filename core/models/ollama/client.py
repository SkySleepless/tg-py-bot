"""
Ollama client for local model inference.
Provides zero-cost AI model access through local Ollama installation.
"""

import logging
from typing import Dict, Any, Optional

from ..base import BaseModelClient, ModelResponse, ModelType


logger = logging.getLogger(__name__)


class OllamaClient(BaseModelClient):
    """Client for Ollama local model inference."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama client.
        
        Args:
            config: Configuration dictionary from BotConfig.model_configs["ollama"]
        """
        super().__init__(config)
        self.model_type = ModelType.OLLAMA
        self.api_key = config.get("api_key", "")
        self.api_url = config.get("api_url", "http://localhost:11434")

    def calculate_cost(self, tokens_used: Dict[str, int]) -> float:
        """
        Calculate cost for Ollama (always zero for local model).
        
        Args:
            tokens_used: Dictionary with 'input_tokens' and 'output_tokens' keys
            
        Returns:
            Always 0.0 for Ollama
        """
        return 0.0
    

    async def test_connection(self) -> bool:
        """
        Test connection to Ollama API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Ensure session is initialized
            if self.session is None:
                await self.initialize()
            
            async with self.session.get(
                f"{self.api_url}/api/tags",
                timeout=5
            ) as response:
                return response.status == 200
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Ollama model.
        
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


# Factory function for creating Ollama client
def create_ollama_client(config: Dict[str, Any]) -> OllamaClient:
    """
    Create an Ollama client instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OllamaClient instance
    """
    return OllamaClient(config)