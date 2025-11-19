"""Gradient accumulation utilities."""

import logging

logger = logging.getLogger(__name__)


class GradientAccumulator:
    """Manages gradient accumulation for simulating larger batch sizes."""
    
    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_update(self, step: int) -> bool:
        """
        Check if optimizer should update at this step.
        
        Args:
            step: Current training step
            
        Returns:
            True if should update, False otherwise
        """
        return step % self.accumulation_steps == 0
    
    def reset(self):
        """Reset accumulation counter."""
        self.current_step = 0
