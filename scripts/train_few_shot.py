#!/usr/bin/env python3
"""Few-shot training script for Cross-Lingual QA models."""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.xquad_loader import XQuADLoader
from src.data.mlqa_loader import MLQALoader
from src.training.few_shot_sampler import FewShotSampler
from src.training.few_shot_trainer import FewShotTrainer
from src.training.experiment_tracker import ExperimentTracker
from src.models.mbert_wrapper import MBERTModelWrapper
from src.models.mt5_wrapper import MT5ModelWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Cross-Lingual QA model with few-shot learning"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['mbert', 'mt5'],
        default='mbert',
        help='Model type to train'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to zero-shot checkpoint'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to multilingual dataset (XQuAD or MLQA)'
    )
    
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['xquad', 'mlqa'],
        default='xquad',
        help='Type of dataset'
    )
    
    parser.add_argument(
        '--num-shots',
        type=int,
        choices=[1, 5, 10, 50],
        default=10,
        help='Number of examples per language pair'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/checkpoints',
        help='Directory to save model checkpoints'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for experiment tracking'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Use Weights & Biases for experiment tracking'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Set random seed to {seed}")


def load_data(data_path: str, dataset_type: str):
    """
    Load multilingual dataset.
    
    Args:
        data_path: Path to dataset
        dataset_type: 'xquad' or 'mlqa'
        
    Returns:
        List of QA examples
    """
    logger.info(f"Loading {dataset_type} data from {data_path}")
    
    if dataset_type == 'xquad':
        loader = XQuADLoader()
    elif dataset_type == 'mlqa':
        loader = MLQALoader()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    examples = loader.load(data_path)
    
    logger.info(f"Loaded {len(examples)} examples")
    
    return examples


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Generate experiment name
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"few_shot_{args.model}_{args.num_shots}shot_{timestamp}"
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb
    )
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'num_shots': args.num_shots,
        'early_stopping_patience': 3,
        'max_grad_norm': 1.0
    }
    
    # Log hyperparameters
    tracker.log_hyperparameters({
        'model_type': args.model,
        'checkpoint': args.checkpoint,
        'dataset_type': args.dataset_type,
        'seed': args.seed,
        **config
    })
    
    # Load data
    all_examples = load_data(args.data_path, args.dataset_type)
    
    # Sample few-shot examples
    sampler = FewShotSampler(num_shots=args.num_shots, seed=args.seed)
    train_examples, val_examples = sampler.sample(all_examples)
    
    logger.info(
        f"Sampled {len(train_examples)} training examples "
        f"and {len(val_examples)} validation examples"
    )
    
    # Load model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    
    if args.model == 'mbert':
        model = MBERTModelWrapper()
    elif args.model == 'mt5':
        model = MT5ModelWrapper()
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Model loaded successfully")
    
    # Create trainer
    trainer = FewShotTrainer(
        model=model,
        train_examples=train_examples,
        val_examples=val_examples,
        output_dir=args.output_dir,
        experiment_tracker=tracker,
        **config
    )
    
    # Train model
    logger.info("Starting few-shot training...")
    
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Finish tracking
        tracker.finish()
    
    logger.info(f"Experiment {args.experiment_name} completed")


if __name__ == "__main__":
    main()
