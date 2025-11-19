"""Model manager for loading and caching QA models."""

import logging
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import psutil

from src.models.base_model import QAModelWrapper
from src.models.mbert_wrapper import MBERTModelWrapper
from src.models.mt5_wrapper import MT5ModelWrapper

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and caching."""
    
    def __init__(
        self,
        cache_dir: str = "models/cache",
        max_memory_gb: float = 8.0,
        device: Optional[str] = None
    ):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for model cache
            max_memory_gb: Maximum memory to use for cached models (GB)
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        
        # Set device
        if device is None:
            self.device = self._get_optimal_device()
        else:
            self.device = torch.device(device)
        
        # Model cache: model_id -> (model, memory_usage)
        self.loaded_models: Dict[str, tuple[QAModelWrapper, int]] = {}
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def load_model(
        self,
        model_type: str,
        checkpoint_path: Optional[str] = None,
        model_id: Optional[str] = None,
        **model_kwargs
    ) -> QAModelWrapper:
        """
        Load model with caching.
        
        Args:
            model_type: Type of model ('mbert' or 'mt5')
            checkpoint_path: Path to model checkpoint (optional)
            model_id: Unique identifier for caching (auto-generated if None)
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded QA model wrapper
        """
        # Generate model ID if not provided
        if model_id is None:
            model_id = self._generate_model_id(model_type, checkpoint_path)
        
        # Check if model is already loaded
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} found in cache")
            return self.loaded_models[model_id][0]
        
        # Check memory before loading
        self._ensure_memory_available()
        
        # Load model
        logger.info(f"Loading model {model_id} (type: {model_type})")
        
        if model_type.lower() == 'mbert':
            model = self._load_mbert(checkpoint_path, **model_kwargs)
        elif model_type.lower() == 'mt5':
            model = self._load_mt5(checkpoint_path, **model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Estimate memory usage
        memory_usage = self._estimate_model_memory(model)
        
        # Cache model
        self.loaded_models[model_id] = (model, memory_usage)
        
        logger.info(
            f"Model {model_id} loaded successfully "
            f"(estimated memory: {memory_usage / 1024 / 1024:.2f} MB)"
        )
        
        return model
    
    def _load_mbert(
        self,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> MBERTModelWrapper:
        """
        Load mBERT model.
        
        Args:
            checkpoint_path: Path to checkpoint
            **kwargs: Additional model arguments
            
        Returns:
            MBERTModelWrapper instance
        """
        model = MBERTModelWrapper(device=str(self.device), **kwargs)
        
        if checkpoint_path:
            model.load(checkpoint_path)
        
        return model
    
    def _load_mt5(
        self,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> MT5ModelWrapper:
        """
        Load mT5 model.
        
        Args:
            checkpoint_path: Path to checkpoint
            **kwargs: Additional model arguments
            
        Returns:
            MT5ModelWrapper instance
        """
        model = MT5ModelWrapper(device=str(self.device), **kwargs)
        
        if checkpoint_path:
            model.load(checkpoint_path)
        
        return model
    
    def unload_model(self, model_id: str):
        """
        Free memory by unloading model.
        
        Args:
            model_id: ID of model to unload
        """
        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} not found in cache")
            return
        
        model, memory_usage = self.loaded_models[model_id]
        
        # Delete model
        del self.loaded_models[model_id]
        del model
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if self.device.type in ['cuda', 'mps']:
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        logger.info(
            f"Model {model_id} unloaded "
            f"(freed ~{memory_usage / 1024 / 1024:.2f} MB)"
        )
    
    def get_model(self, model_id: str) -> Optional[QAModelWrapper]:
        """
        Get cached model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model if cached, None otherwise
        """
        if model_id in self.loaded_models:
            return self.loaded_models[model_id][0]
        return None
    
    def list_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all loaded models.
        
        Returns:
            Dictionary of model info
        """
        models_info = {}
        
        for model_id, (model, memory_usage) in self.loaded_models.items():
            models_info[model_id] = {
                'type': type(model).__name__,
                'memory_mb': memory_usage / 1024 / 1024,
                'device': str(self.device)
            }
        
        return models_info
    
    def _generate_model_id(
        self,
        model_type: str,
        checkpoint_path: Optional[str]
    ) -> str:
        """
        Generate unique model ID.
        
        Args:
            model_type: Type of model
            checkpoint_path: Path to checkpoint
            
        Returns:
            Model ID string
        """
        if checkpoint_path:
            checkpoint_name = Path(checkpoint_path).stem
            return f"{model_type}_{checkpoint_name}"
        return f"{model_type}_pretrained"
    
    def _estimate_model_memory(self, model: QAModelWrapper) -> int:
        """
        Estimate model memory usage.
        
        Args:
            model: Model wrapper
            
        Returns:
            Estimated memory in bytes
        """
        # Get model parameters
        if hasattr(model, 'model'):
            params = sum(p.numel() * p.element_size() 
                        for p in model.model.parameters())
            buffers = sum(b.numel() * b.element_size() 
                         for b in model.model.buffers())
        else:
            params = 0
            buffers = 0
        
        # Add overhead (approximate)
        overhead = int((params + buffers) * 0.2)
        
        return params + buffers + overhead
    
    def _ensure_memory_available(self):
        """Ensure sufficient memory is available."""
        current_usage = sum(mem for _, mem in self.loaded_models.values())
        
        if current_usage > self.max_memory_bytes * 0.8:
            logger.warning(
                f"Memory usage high ({current_usage / 1024 / 1024 / 1024:.2f} GB). "
                "Consider unloading models."
            )
        
        # Get system memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logger.warning(
                f"System memory usage critical: {memory.percent}%. "
                "Unloading least recently used model."
            )
            # Unload first model (simple LRU)
            if self.loaded_models:
                first_model_id = next(iter(self.loaded_models))
                self.unload_model(first_model_id)
    
    def clear_cache(self):
        """Clear all cached models."""
        model_ids = list(self.loaded_models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
        
        logger.info("Model cache cleared")
