"""
Logging configuration for RAG system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig


def setup_logging(config: Optional[DictConfig] = None, 
                 log_level: Optional[str] = None,
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for RAG system.
    
    Args:
        config: RAG configuration (optional)
        log_level: Logging level (overrides config)
        log_file: Log file path (overrides config)
    
    Returns:
        Configured logger
    """
    # Get logging configuration
    if config is not None and hasattr(config, 'logging'):
        level = log_level or config.logging.get('level', 'INFO')
        format_str = config.logging.get('format', 
                                       '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_path = log_file or config.logging.get('log_file', None)
    else:
        level = log_level or 'INFO'
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_path = log_file
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if file_path:
        # Create log directory if it doesn't exist
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create RAG-specific logger
    rag_logger = logging.getLogger('rag')
    rag_logger.setLevel(numeric_level)
    
    return rag_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific RAG component.
    
    Args:
        name: Logger name (e.g., 'rag.retrieval', 'rag.generation')
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f'rag.{name}')


class LoggerMixin:
    """Mixin class to add logging capability to RAG components."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this component."""
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__
            self._logger = get_logger(class_name.lower())
        return self._logger
