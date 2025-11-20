"""
Configuration management for RAG system using Hydra.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


class RAGConfig:
    """RAG configuration manager using Hydra."""
    
    def __init__(self, config_name: str = "default", config_path: Optional[str] = None):
        """
        Initialize RAG configuration.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            config_path: Path to config directory (defaults to configs/rag/)
        """
        if config_path is None:
            # Get absolute path to configs/rag directory
            project_root = Path(__file__).parent.parent.parent
            config_path = str(project_root / "configs" / "rag")
        
        self.config_path = config_path
        self.config_name = config_name
        self._config: Optional[DictConfig] = None
    
    def load(self, overrides: Optional[list] = None) -> DictConfig:
        """
        Load configuration with optional overrides.
        
        Args:
            overrides: List of config overrides (e.g., ["retrieval.top_k=10"])
        
        Returns:
            Loaded configuration
        """
        if overrides is None:
            overrides = []
        
        # Initialize Hydra with absolute config path
        with initialize_config_dir(config_dir=self.config_path, version_base=None):
            self._config = compose(config_name=self.config_name, overrides=overrides)
        
        return self._config
    
    @property
    def config(self) -> DictConfig:
        """Get loaded configuration."""
        if self._config is None:
            self.load()
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Dot-separated key (e.g., "retrieval.top_k")
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self.config, resolve=True)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return OmegaConf.to_yaml(self.config)


def load_rag_config(config_name: str = "default", 
                    overrides: Optional[list] = None) -> DictConfig:
    """
    Convenience function to load RAG configuration.
    
    Args:
        config_name: Name of the config file
        overrides: List of config overrides
    
    Returns:
        Loaded configuration
    """
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()
    
    config_manager = RAGConfig(config_name=config_name)
    config = config_manager.load(overrides=overrides)
    
    # Resolve environment variables
    OmegaConf.resolve(config)
    
    # Inject OpenAI API key from environment if using OpenAI
    if config.get('generator', {}).get('type') == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and 'generator' in config:
            config.generator.api_key = api_key
            print(f"✅ OpenAI API key injected into config: {api_key[:20]}...")
        else:
            print("❌ OpenAI generator configured but no API key found!")
    
    return config
