"""
Configuration management
"""

import os
from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    api_key: str = ""
    api_url: str = ""
    model: str = ""
    timeout: int = 60
    max_tokens: int = 4096
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    priority: int = 1
    best_for: List[str] = []
    enabled: bool = True

class BotConfig(BaseSettings):
    """Main configuration for the Dynamic Agent Bot."""
    
    # Telegram Bot Configuration
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_bot_admin: str = Field(default="", alias="ADMIN_USER_IDS")
    telegram_bot_user: str = Field(default="", alias="VALID_USER_IDS")

    # Ollama Configuration
    ollama_api_key: Optional[str] = Field(default=None, alias="OLLAMA_API_KEY")
    ollama_api_url: str = Field(default="http://localhost:11434", alias="OLLAMA_API_URL")
    ollama_model: str = Field(default="hf.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF:latest", alias="OLLAMA_MODEL")
    
    # Mistral API Configuration
    mistral_api_key: Optional[str] = Field(default=None, alias="MISTRAL_API_KEY")
    mistral_api_url: str = Field(default="https://api.mistral.ai", alias="MISTRAL_API_URL")
    mistral_model: str = Field(default="mistral-small-latest", alias="MISTRAL_MODEL")

    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # Model configurations
    model_configs: Dict[str, Any] = {}
    
    @field_validator('telegram_bot_token')
    @classmethod
    def validate_required_keys(cls, v: str, info: ValidationInfo) -> str:
        if not v:
            raise ValueError(f"{info.field_name} is required")
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_model_configs()
    
    def _initialize_model_configs(self):
        """Initialize model configurations."""
        self.model_configs = {
            "mistral": {
                "api_key": self.mistral_api_key,
                "api_url": self.mistral_api_url,
                "model": self.mistral_model,
                "timeout": 60,
                "max_tokens": 131072, #128k
                "input_cost_per_token": 0.0000004,
                "output_cost_per_token": 0.000002,
                "priority": 1,  # Primary model for all tasks
                "best_for": ["intent_classification", "direct_response", "agent_config_generation",
                           "test_case_generation", "description_optimization", "multi_turn_dialogue",
                           "knowledge_sync", "complex_logic", "high_precision_tasks"],
                "enabled": self.mistral_api_key is not None
            },
            "ollama": {
                "api_key": self.ollama_api_key,
                "api_url": self.ollama_api_url,
                "model": self.ollama_model,
                "timeout": 30,
                "max_tokens": 32000,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "priority": 2,  # Fallback only when Mistral is unavailable
                "best_for": ["intent_classification", "simple_response", "knowledge_sync"],
                "enabled": True
            }
        }

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        for model_type, config in self.model_configs.items():
            if model_type == model_name:
                return config
        return None
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled."""
        config = self.get_model_config(model_name)
        if config:
            return config.get("enabled", False)
        return False