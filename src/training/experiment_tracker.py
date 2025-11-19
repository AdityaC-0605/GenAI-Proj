"""Experiment tracking module for Cross-Lingual QA training."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import platform
import torch

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track experiments with hyperparameters, metrics, and system info."""
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "experiments/tracking",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save tracking data
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.wandb_run = None
        
        # Initialize experiment data
        self.experiment_data = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'system_info': self._collect_system_info(),
            'hyperparameters': {},
            'model_config': {},
            'metrics': {
                'train': [],
                'validation': []
            },
            'checkpoints': []
        }
        
        # Initialize W&B if requested
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or "cross-lingual-qa",
                    name=experiment_name,
                    config={}
                )
                logger.info(f"Initialized W&B tracking for {experiment_name}")
            except ImportError:
                logger.warning("wandb not installed, falling back to local tracking only")
                self.use_wandb = False
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        self.experiment_data['hyperparameters'].update(hyperparameters)
        
        logger.info(f"Logged hyperparameters: {hyperparameters}")
        
        if self.use_wandb and self.wandb_run:
            self.wandb_run.config.update(hyperparameters)
    
    def log_model_config(self, model_config: Dict[str, Any]):
        """
        Log model architecture details.
        
        Args:
            model_config: Dictionary of model configuration
        """
        self.experiment_data['model_config'].update(model_config)
        
        logger.info(f"Logged model config: {model_config}")
        
        if self.use_wandb and self.wandb_run:
            self.wandb_run.config.update({'model': model_config})
    
    def log_metrics(
        self,
        epoch: int,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log training and validation metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        if train_metrics:
            train_entry = {'epoch': epoch, **train_metrics}
            self.experiment_data['metrics']['train'].append(train_entry)
            
            logger.info(f"Epoch {epoch} - Train metrics: {train_metrics}")
            
            if self.use_wandb and self.wandb_run:
                self.wandb_run.log({f'train/{k}': v for k, v in train_metrics.items()}, step=epoch)
        
        if val_metrics:
            val_entry = {'epoch': epoch, **val_metrics}
            self.experiment_data['metrics']['validation'].append(val_entry)
            
            logger.info(f"Epoch {epoch} - Val metrics: {val_metrics}")
            
            if self.use_wandb and self.wandb_run:
                self.wandb_run.log({f'val/{k}': v for k, v in val_metrics.items()}, step=epoch)
    
    def log_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Log checkpoint information.
        
        Args:
            checkpoint_path: Path to checkpoint file
            epoch: Epoch number
            metrics: Metrics at checkpoint
            is_best: Whether this is the best checkpoint
        """
        checkpoint_info = {
            'path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat()
        }
        
        self.experiment_data['checkpoints'].append(checkpoint_info)
        
        logger.info(f"Logged checkpoint: {checkpoint_path} (epoch {epoch}, best={is_best})")
        
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log({
                'checkpoint_epoch': epoch,
                'checkpoint_is_best': is_best
            })
    
    def save(self):
        """Save experiment data to disk."""
        self.experiment_data['end_time'] = datetime.now().isoformat()
        
        # Save to JSON
        output_path = self.output_dir / f"{self.experiment_name}.json"
        with open(output_path, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
        
        logger.info(f"Experiment data saved to {output_path}")
    
    def finish(self):
        """Finish experiment tracking."""
        self.save()
        
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
            logger.info("Finished W&B tracking")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collect system information.
        
        Returns:
            Dictionary of system information
        """
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        # Get device name
        if torch.cuda.is_available():
            system_info['device_name'] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            system_info['device_name'] = 'Apple Silicon (MPS)'
        else:
            system_info['device_name'] = 'CPU'
        
        return system_info
    
    def get_metrics_history(self, split: str = 'train') -> List[Dict[str, Any]]:
        """
        Get metrics history for a split.
        
        Args:
            split: 'train' or 'validation'
            
        Returns:
            List of metric dictionaries
        """
        return self.experiment_data['metrics'].get(split, [])
    
    def get_best_checkpoint(self, metric: str = 'loss', mode: str = 'min') -> Optional[Dict[str, Any]]:
        """
        Get best checkpoint based on a metric.
        
        Args:
            metric: Metric name to compare
            mode: 'min' or 'max'
            
        Returns:
            Best checkpoint info or None
        """
        checkpoints = self.experiment_data['checkpoints']
        
        if not checkpoints:
            return None
        
        if mode == 'min':
            best_checkpoint = min(
                checkpoints,
                key=lambda x: x['metrics'].get(metric, float('inf'))
            )
        else:
            best_checkpoint = max(
                checkpoints,
                key=lambda x: x['metrics'].get(metric, float('-inf'))
            )
        
        return best_checkpoint


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only best checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        self.checkpoints = []
        self.best_metric = None
        self.best_checkpoint_path = None
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Current metrics
            config: Training configuration
            filename: Optional custom filename
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Track checkpoint
        self.checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics
        })
        
        # Cleanup old checkpoints
        if not self.save_best_only:
            self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def save_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        metric_name: str = 'val_loss',
        mode: str = 'min'
    ) -> Optional[str]:
        """
        Save checkpoint if it's the best so far.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Current metrics
            config: Training configuration
            metric_name: Metric to compare
            mode: 'min' or 'max'
            
        Returns:
            Path to saved checkpoint or None if not best
        """
        current_metric = metrics.get(metric_name)
        
        if current_metric is None:
            logger.warning(f"Metric {metric_name} not found in metrics")
            return None
        
        is_best = False
        
        if self.best_metric is None:
            is_best = True
        elif mode == 'min' and current_metric < self.best_metric:
            is_best = True
        elif mode == 'max' and current_metric > self.best_metric:
            is_best = True
        
        if is_best:
            self.best_metric = current_metric
            
            checkpoint_path = self.save_checkpoint(
                model, optimizer, epoch, metrics, config,
                filename="best_checkpoint.pt"
            )
            
            self.best_checkpoint_path = checkpoint_path
            
            logger.info(f"New best checkpoint: {metric_name}={current_metric:.4f}")
            
            return checkpoint_path
        
        return None
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics'),
            'config': checkpoint.get('config')
        }
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by epoch
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x['epoch'])
        
        # Remove oldest checkpoints
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint['path'])
            
            # Don't remove best checkpoint
            if str(checkpoint_path) == self.best_checkpoint_path:
                continue
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
        
        # Update checkpoint list
        self.checkpoints = sorted_checkpoints[-self.max_checkpoints:]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x['epoch'])
        return latest['path']
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint or None
        """
        return self.best_checkpoint_path
