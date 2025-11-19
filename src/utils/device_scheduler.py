"""Device scheduler for Apple Silicon optimization."""

import logging
import torch
import psutil

logger = logging.getLogger(__name__)


class DeviceScheduler:
    """Manages device placement for Apple Silicon with automatic fallback."""
    
    def __init__(self):
        """Initialize device scheduler."""
        self.device = self._get_optimal_device()
        self.unsupported_ops = set()
        
        logger.info(f"DeviceScheduler initialized with device: {self.device}")
        
        # Monitor memory
        self._log_memory_info()
    
    def _get_optimal_device(self) -> torch.device:
        """
        Get optimal device (MPS > CUDA > CPU priority).
        
        Returns:
            Optimal torch device
        """
        if torch.backends.mps.is_available():
            logger.info("MPS backend available, using Apple Silicon GPU")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            logger.info("Using CPU")
            return torch.device("cpu")
    
    def to_device(self, tensor: torch.Tensor, operation_name: str = None) -> torch.Tensor:
        """
        Move tensor to appropriate device with fallback.
        
        Args:
            tensor: Tensor to move
            operation_name: Name of operation (for tracking unsupported ops)
            
        Returns:
            Tensor on appropriate device
        """
        try:
            return tensor.to(self.device)
        except RuntimeError as e:
            if "mps" in str(e).lower():
                # Fallback to CPU for unsupported operations
                if operation_name:
                    if operation_name not in self.unsupported_ops:
                        logger.warning(
                            f"Operation '{operation_name}' not supported on MPS, "
                            f"falling back to CPU"
                        )
                        self.unsupported_ops.add(operation_name)
                return tensor.to("cpu")
            raise
    
    def get_memory_usage(self) -> dict:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        memory_info = {
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        if self.device.type == 'cuda':
            memory_info['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        return memory_info
    
    def check_memory_warning(self) -> bool:
        """
        Check if memory usage exceeds warning threshold.
        
        Returns:
            True if warning threshold exceeded
        """
        memory = psutil.virtual_memory()
        
        if memory.percent > 80:
            logger.warning(
                f"Unified memory usage high: {memory.percent}% "
                f"({memory.available / (1024**3):.2f} GB available)"
            )
            return True
        
        return False
    
    def _log_memory_info(self):
        """Log current memory information."""
        memory = psutil.virtual_memory()
        logger.info(
            f"System memory: {memory.total / (1024**3):.2f} GB total, "
            f"{memory.available / (1024**3):.2f} GB available ({memory.percent}% used)"
        )
        
        if self.device.type == 'cuda':
            logger.info(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
            )
