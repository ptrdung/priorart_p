"""
Configuration settings for the AI agent
Contains all configuration variables and API keys
"""

import os
from typing import Dict, Any

class Settings:
    """Configuration settings for the patent AI agent"""
    
    # API Keys
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "BSAQlxb-jIHFbW1mK0_S4zlTqfkuA3Z")
    
    # Model Configuration
    DEFAULT_MODEL_NAME = "qwen2.5:14b-instruct"
    MODEL_TEMPERATURE = 0.7
    
    # Search Configuration
    MAX_SEARCH_RESULTS = 5
    MAX_SNIPPETS = 3
    
    # IPC Classification API
    IPC_API_URL = "https://ipccat.wipo.int/EN/query"
    IPC_PREDICTIONS_COUNT = 3
    IPC_HIERARCHIC_LEVEL = "SUBGROUP"
    
    # Brave Search API
    BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
    
    # File Paths
    OUTPUT_DIR = "outputs"
    LOGS_DIR = "logs"
    
    # Workflow Configuration
    USE_CHECKPOINTER = False
    THREAD_ID = "extraction_thread_1"
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        return {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        return {
            "TAVILY_API_KEY": bool(cls.TAVILY_API_KEY),
            "BRAVE_API_KEY": bool(cls.BRAVE_API_KEY)
        }

# Create global settings instance
settings = Settings()
