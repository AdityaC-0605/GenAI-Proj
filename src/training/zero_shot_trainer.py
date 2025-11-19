"""Zero-shot training pipeline for Cross-Lingual QA models."""

import os
import json
import math
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

from src.models.base_model import QAModelWrapper
from src.utils.gradient_accumulation import GradientAccumulator
from src.utils.mixed_precision import MixedPrecisionManager

logger = logging.getLogger(__name__)


class ZeroShotTrainer:
    """Trainer for zero-shot cross-lingual QA models."""
    
    def __init__(
        self,
        model: QAModelWrapper,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        learning_rate: float = 3e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 3,
        checkpoint_dir: str = "models/checkpoints",
        use_mixed_precision: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize zero-shot trainer.
        
        Args:
            model: QA model wrapper
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            warmup_ratio: Ratio of warmup steps
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            use_mixed_precision: Whether to use mixed precision training
            device: Device to use for training
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = self._get_optimal_device()
        else:
            self.device = device
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Calculate total training steps
        self.total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self.warmup_steps = int(self.total_steps * warmup_ratio)
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Initialize gradient accumulator
        self.grad_accumulator = GradientAccumulator(gradient_accumulation_steps)
        
        # Initialize mixed precision manager
        # Disable mixed precision for mT5 on MPS (causes NaN issues)
        model_type = ''
        if hasattr(model, 'model_name'):
            model_type = model.model_name
        elif hasattr(model, 'model') and hasattr(model.model, 'model_name'):
            model_type = model.model.model_name
        elif 'mt5' in str(type(model)).lower() or 't5' in str(type(model)).lower():
            model_type = 'mt5'
        
        if 'mt5' in model_type.lower() and self.device.type == 'mps':
            logger.warning("Disabling mixed precision for mT5 on MPS to avoid NaN issues")
            self.use_mixed_precision = False
        else:
            self.use_mixed_precision = use_mixed_precision and self.device.type in ['mps', 'cuda']
        
        if self.use_mixed_precision:
            self.mixed_precision_manager = MixedPrecisionManager(device=self.device)
        else:
            self.mixed_precision_manager = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Get model parameters
        if hasattr(self.model, 'model'):
            model_params = self.model.model.parameters()
        else:
            model_params = self.model.parameters()
        
        return AdamW(
            model_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

    def train(self) -> Dict[str, Any]:
        """
        Run the complete training loop.
        
        Returns:
            Dictionary containing training history and final metrics
        """
        logger.info(f"Starting zero-shot training on {self.device}")
        logger.info(f"Total epochs: {self.num_epochs}, Total steps: {self.total_steps}")
        logger.info(f"Warmup steps: {self.warmup_steps}, Gradient accumulation: {self.gradient_accumulation_steps}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self._train_epoch()
            
            # Validate
            val_loss = self._validate()
            
            # Log metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_best_model()
            else:
                self.epochs_without_improvement += 1
                
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("Training completed!")
        return {
            'history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1
        }
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Perform training step
            if self.use_mixed_precision:
                loss = self._train_step_mixed_precision(batch)
            else:
                loss = self._train_step_standard(batch)
            
            # Skip NaN losses
            if isinstance(loss, torch.Tensor):
                if torch.isnan(loss):
                    logger.warning(f"NaN loss at batch {batch_idx + 1}, skipping...")
                    continue
            elif isinstance(loss, (int, float)):
                if math.isnan(loss):
                    logger.warning(f"NaN loss at batch {batch_idx + 1}, skipping...")
                    continue
            
            total_loss += loss
            num_batches += 1
            
            # Gradient accumulation
            if self.grad_accumulator.should_update(batch_idx + 1):
                # Clip gradients (unscale first if using mixed precision)
                if hasattr(self.model, 'model'):
                    if self.use_mixed_precision and self.mixed_precision_manager and self.mixed_precision_manager.scaler:
                        self.mixed_precision_manager.unscale_gradients(self.optimizer)
                    
                    # Check for NaN/inf gradients before clipping
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.max_grad_norm
                    )
                    
                    # Check if gradients are valid
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        logger.warning(f"Invalid gradient norm detected: {total_norm}. Skipping optimizer step.")
                        self.optimizer.zero_grad()
                        continue
                
                # Optimizer step
                if self.use_mixed_precision and self.mixed_precision_manager:
                    self.mixed_precision_manager.step(self.optimizer)
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Log progress
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                # Check for NaN before logging
                if torch.isnan(torch.tensor(avg_loss)) or str(avg_loss) == 'nan':
                    logger.warning(
                        f"Epoch {self.current_epoch + 1}, Batch {batch_idx + 1}/{len(self.train_dataloader)}, "
                        f"Loss: nan (training may be unstable)"
                    )
                else:
                    logger.info(
                        f"Epoch {self.current_epoch + 1}, Batch {batch_idx + 1}/{len(self.train_dataloader)}, "
                        f"Loss: {avg_loss:.4f}"
                    )
        
        return total_loss / num_batches
    
    def _train_step_standard(self, batch: Dict) -> float:
        """
        Perform a standard training step without mixed precision.
        
        Args:
            batch: Batch data
            
        Returns:
            Loss value
        """
        # Move batch to device
        batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items()}
        
        # Forward pass - get loss tensor directly
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'forward'):
            outputs = self.model.model(**batch_device)
            if hasattr(outputs, 'loss'):
                loss_tensor = outputs.loss
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                loss_tensor = outputs[0]
            else:
                raise ValueError("Could not extract loss from model output")
        else:
            # Fallback to train_step if model structure is different
            loss_value = self.model.train_step(batch)
            return loss_value
        
        # Check for NaN
        if torch.isnan(loss_tensor):
            logger.warning("NaN loss detected! Skipping backward pass.")
            return float('nan')
        
        # Scale loss for gradient accumulation
        scaled_loss_tensor = loss_tensor / self.gradient_accumulation_steps
        
        # Backward pass
        scaled_loss_tensor.backward()
        
        # Return loss value for logging
        return loss_tensor.item()
    
    def _train_step_mixed_precision(self, batch: Dict) -> float:
        """
        Perform a training step with mixed precision.
        
        Args:
            batch: Batch data
            
        Returns:
            Loss value
        """
        # Move batch to device
        batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items()}
        
        # Forward pass with mixed precision - get loss tensor directly
        with self.mixed_precision_manager.autocast():
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'forward'):
                outputs = self.model.model(**batch_device)
                if hasattr(outputs, 'loss'):
                    loss_tensor = outputs.loss
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    loss_tensor = outputs[0]
                else:
                    raise ValueError("Could not extract loss from model output")
            else:
                # Fallback
                loss_value = self.model.train_step(batch)
                return loss_value
        
        # Check for NaN
        if torch.isnan(loss_tensor):
            logger.warning("NaN loss detected! Skipping backward pass.")
            return float('nan')
        
        # Scale loss for gradient accumulation
        scaled_loss_tensor = loss_tensor / self.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        self.mixed_precision_manager.backward(scaled_loss_tensor)
        
        # Return loss value for logging
        return loss_tensor.item()
    
    def _validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                loss = self.model.train_step(batch)
                total_loss += loss
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.model.state_dict() if hasattr(self.model, 'model') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'training_history': self.training_history,
            'config': {
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'warmup_ratio': self.warmup_ratio,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'max_grad_norm': self.max_grad_norm
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _save_best_model(self):
        """Save the best model based on validation loss."""
        best_model_path = self.checkpoint_dir / "best_model.pt"
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.model.state_dict() if hasattr(self.model, 'model') else self.model.state_dict(),
            'val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, best_model_path)
        logger.info(f"Best model saved to {best_model_path} with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if hasattr(self.model, 'model'):
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
