"""Mixed precision training utilities."""

import logging
import torch

logger = logging.getLogger(__name__)


class MixedPrecisionManager:
    """Manages mixed precision training with automatic fallback."""
    
    def __init__(self, device: torch.device):
        """
        Initialize mixed precision manager.
        
        Args:
            device: Device being used for training
        """
        self.device = device
        self.enabled = device.type in ['cuda', 'mps']
        
        if self.enabled:
            if device.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Mixed precision enabled with CUDA")
            elif device.type == 'mps':
                # MPS has limited mixed precision support
                self.scaler = None
                logger.info("Mixed precision enabled with MPS (limited support)")
        else:
            self.scaler = None
            logger.info("Mixed precision disabled (CPU mode)")
    
    def autocast(self):
        """
        Get autocast context manager.
        
        Returns:
            Autocast context manager
        """
        if self.device.type == 'cuda':
            return torch.cuda.amp.autocast()
        elif self.device.type == 'mps':
            # MPS autocast (if available in PyTorch version)
            try:
                return torch.autocast(device_type='mps', dtype=torch.float16)
            except:
                # Fallback to no-op context
                return torch.no_grad().__class__()
        else:
            # No-op context for CPU
            return torch.no_grad().__class__()
    
    def backward(self, loss: torch.Tensor):
        """
        Perform backward pass with gradient scaling.
        
        Args:
            loss: Loss tensor
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Perform optimizer step with unscaling.
        
        Args:
            optimizer: Optimizer instance
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """
        Unscale gradients before clipping.
        
        Args:
            optimizer: Optimizer instance
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
