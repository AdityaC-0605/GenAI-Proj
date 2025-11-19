#!/usr/bin/env python3
"""Zero-shot training script for Cross-Lingual QA models."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress urllib3 warnings (harmless on macOS)
try:
    from src.utils.warning_suppressor import suppress_urllib3_warnings
    suppress_urllib3_warnings()
except ImportError:
    pass

from src.data.squad_loader import SQuADLoader
from src.data.data_splitter import DataSplitter
from src.models.mbert_wrapper import MBERTModelWrapper
from src.models.mt5_wrapper import MT5ModelWrapper
from src.training.zero_shot_trainer import ZeroShotTrainer
from src.training.experiment_tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Cross-Lingual QA model with zero-shot learning"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['mbert', 'mt5'],
        default='mbert',
        help='Model type to train'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to SQuAD dataset'
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
        default=None,
        help='Training batch size (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Use Weights & Biases for experiment tracking'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='cross-lingual-qa',
        help='W&B project name'
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


def load_data(data_path: str, split_ratio: tuple = (0.8, 0.1, 0.1)):
    """
    Load and split training data.
    
    Args:
        data_path: Path to dataset
        split_ratio: Train/val/test split ratio
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load SQuAD dataset
    loader = SQuADLoader()
    examples = loader.load(data_path)
    
    logger.info(f"Loaded {len(examples)} examples")
    
    # Split data
    splitter = DataSplitter(
        train_ratio=split_ratio[0],
        val_ratio=split_ratio[1],
        test_ratio=split_ratio[2]
    )
    train_examples, val_examples, test_examples = splitter.split(examples, stratify_by_language=False)
    
    logger.info(
        f"Split data: train={len(train_examples)}, "
        f"val={len(val_examples)}, test={len(test_examples)}"
    )
    
    return train_examples, val_examples, test_examples


def create_model(model_type: str, config: dict):
    """
    Create model wrapper.
    
    Args:
        model_type: 'mbert' or 'mt5'
        config: Model configuration
        
    Returns:
        Model wrapper instance
    """
    logger.info(f"Creating {model_type} model")
    
    if model_type == 'mbert':
        model = MBERTModelWrapper(
            model_name=config.get('model_name', 'bert-base-multilingual-cased'),
            max_seq_length=config.get('max_seq_length', 384),
            doc_stride=config.get('doc_stride', 128)
        )
    elif model_type == 'mt5':
        model = MT5ModelWrapper(
            model_name=config.get('model_name', 'google/mt5-base'),
            max_input_length=config.get('max_seq_length', 512),
            max_output_length=config.get('max_answer_length', 50)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Model created successfully")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Generate experiment name
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"zero_shot_{args.model}_{timestamp}"
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Load configuration
    # For simplicity, using default config values
    # Use lower learning rate for mT5 to avoid NaN issues
    default_lr = 1e-5 if args.model == 'mt5' else 3e-5
    default_batch_size = 4 if args.model == 'mt5' else 16
    
    config = {
        'model_name': 'bert-base-multilingual-cased' if args.model == 'mbert' else 'google/mt5-base',
        'max_seq_length': 384,
        'doc_stride': 128,
        'max_answer_length': 50,
        'batch_size': args.batch_size or default_batch_size,
        'learning_rate': args.learning_rate or default_lr,
        'num_epochs': args.num_epochs or 3,
        'warmup_ratio': 0.1,
        'gradient_accumulation_steps': 4,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 3
    }
    
    # Log hyperparameters
    tracker.log_hyperparameters({
        'model_type': args.model,
        'seed': args.seed,
        **config
    })
    
    # Load data
    train_examples, val_examples, test_examples = load_data(args.data_path)
    
    # Create model
    model = create_model(args.model, config)
    
    # Log model config
    tracker.log_model_config({
        'model_type': args.model,
        'model_name': config['model_name'],
        'max_seq_length': config['max_seq_length']
    })
    
    # Create data loaders
    from torch.utils.data import DataLoader, Dataset
    
    class MBERTDataset(Dataset):
        """Dataset for mBERT (extractive QA)."""
        def __init__(self, examples, tokenizer, max_length=384):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            example = self.examples[idx]
            
            # Tokenize question and context
            encoding = self.tokenizer(
                example.question,
                example.context,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Get answer positions
            answer_start = example.answers[0].start_position if example.answers else 0
            answer_end = example.answers[0].end_position if example.answers else 0
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
                'start_positions': torch.tensor(answer_start, dtype=torch.long),
                'end_positions': torch.tensor(answer_end, dtype=torch.long)
            }
    
    class MT5Dataset(Dataset):
        """Dataset for mT5 (generative QA)."""
        def __init__(self, examples, tokenizer, max_input_length=512, max_output_length=50):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_input_length = max_input_length
            self.max_output_length = max_output_length
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            example = self.examples[idx]
            
            # Format input as "question: <q> context: <c>"
            input_text = f"question: {example.question} context: {example.context}"
            
            # Get answer text
            answer_text = example.answers[0].text if example.answers else ""
            
            # Tokenize input
            input_encoding = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize target (answer)
            target_encoding = self.tokenizer(
                answer_text,
                max_length=self.max_output_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': input_encoding['input_ids'].squeeze(0),
                'attention_mask': input_encoding['attention_mask'].squeeze(0),
                'labels': target_encoding['input_ids'].squeeze(0)
            }
    
    # Get tokenizer from model
    tokenizer = model.tokenizer
    
    # Create appropriate dataset based on model type
    if args.model == 'mbert':
        train_dataset = MBERTDataset(train_examples, tokenizer, config['max_seq_length'])
        val_dataset = MBERTDataset(val_examples, tokenizer, config['max_seq_length'])
    else:  # mt5
        train_dataset = MT5Dataset(train_examples, tokenizer, config['max_seq_length'], config['max_answer_length'])
        val_dataset = MT5Dataset(val_examples, tokenizer, config['max_seq_length'], config['max_answer_length'])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Create trainer
    trainer = ZeroShotTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        warmup_ratio=config['warmup_ratio'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        max_grad_norm=config['max_grad_norm'],
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_dir=args.output_dir
    )
    
    # Train model
    logger.info("Starting training...")
    
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
