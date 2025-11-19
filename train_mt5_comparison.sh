#!/bin/bash
# Quick mT5 training for comparison with mBERT
# Uses minimal data and epochs for fast training

echo "🚀 Quick mT5 Training for Comparison"
echo "====================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    exit 1
fi

DATA_PATH=${1:-data/squad/train-v2.0.json}

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Data file not found: $DATA_PATH"
    exit 1
fi

echo "📋 Minimal Training Configuration:"
echo "   Model: mT5"
echo "   Data: ~1,500 training examples (minimal subset)"
echo "   Batch Size: 2 (reduced for MPS memory)"
echo "   Gradient Accumulation: 8 (effective batch: 16)"
echo "   Epochs: 1"
echo "   Learning Rate: 3e-5"
echo "   Expected Time: 20-40 minutes"
echo ""

mkdir -p logs models/checkpoints

LOG_FILE="logs/training_mt5_comparison_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging to: $LOG_FILE"
echo ""
echo "☕ Keeping Mac awake..."
echo ""

# Create a minimal training script with memory optimizations
cat > /tmp/train_mt5_minimal.py << 'PYTHON_SCRIPT'
import sys
import os
sys.path.insert(0, '.')

# Set MPS memory limit to avoid OOM (allow more memory usage)
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

from scripts.train_zero_shot import *
import random

# Monkey patch the load_data function to use minimal subset
original_load_data = load_data

def load_data_minimal(data_path, split_ratio=(0.8, 0.1, 0.1)):
    train, val, test = original_load_data(data_path, split_ratio)
    
    # Use only ~1,500 training examples for quick comparison
    max_train_size = 1500
    if len(train) > max_train_size:
        random.seed(42)
        train_subset = random.sample(train, max_train_size)
        print(f"Using minimal subset: {len(train_subset)} training examples (from {len(train)})")
    else:
        train_subset = train
        print(f"Using all available: {len(train_subset)} training examples")
    
    # Also limit validation to speed up
    max_val_size = 200
    if len(val) > max_val_size:
        random.seed(42)
        val_subset = random.sample(val, max_val_size)
        print(f"Using validation subset: {len(val_subset)} examples (from {len(val)})")
    else:
        val_subset = val
    
    return train_subset, val_subset, test

# Replace function
import scripts.train_zero_shot as train_module
train_module.load_data = load_data_minimal

# Patch the config to use gradient_accumulation_steps=8 for batch_size=2
original_main = train_module.main
def patched_main():
    args = train_module.parse_args()
    
    # Set seed
    train_module.set_seed(args.seed)
    
    # Generate experiment name
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"zero_shot_{args.model}_{timestamp}"
    
    logger = train_module.logger
    logger.info(f"Starting experiment: {args.experiment_name}")
    
    # Initialize experiment tracker
    from src.training.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Load configuration with adjusted gradient accumulation
    batch_size = args.batch_size or 2
    gradient_accumulation = 8 if batch_size == 2 else 4  # Higher grad acc for smaller batch
    
    config = {
        'model_name': 'bert-base-multilingual-cased' if args.model == 'mbert' else 'google/mt5-base',
        'max_seq_length': 384,
        'doc_stride': 128,
        'max_answer_length': 50,
        'batch_size': batch_size,
        'learning_rate': args.learning_rate or 3e-5,
        'num_epochs': args.num_epochs or 1,
        'warmup_ratio': 0.1,
        'gradient_accumulation_steps': gradient_accumulation,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 3
    }
    
    # Log hyperparameters
    tracker.log_hyperparameters({
        'model_type': args.model,
        'seed': args.seed,
        **config
    })
    
    # Load data (will use our patched function)
    train_examples, val_examples, test_examples = train_module.load_data(args.data_path)
    
    # Create model
    model = train_module.create_model(args.model, config)
    
    # Log model config
    tracker.log_model_config({
        'model_type': args.model,
        'model_name': config['model_name'],
        'max_seq_length': config['max_seq_length']
    })
    
    # Create data loaders (rest of the original main function)
    import torch
    from torch.utils.data import DataLoader, Dataset
    
    class MBERTDataset(Dataset):
        def __init__(self, examples, tokenizer, max_length=384):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            example = self.examples[idx]
            encoding = self.tokenizer(
                example.question,
                example.context,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
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
        def __init__(self, examples, tokenizer, max_input_length=512, max_output_length=50):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_input_length = max_input_length
            self.max_output_length = max_output_length
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            example = self.examples[idx]
            input_text = f"question: {example.question} context: {example.context}"
            answer_text = example.answers[0].text if example.answers else ""
            input_encoding = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
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
    
    tokenizer = model.tokenizer
    
    if args.model == 'mbert':
        train_dataset = MBERTDataset(train_examples, tokenizer, config['max_seq_length'])
        val_dataset = MBERTDataset(val_examples, tokenizer, config['max_seq_length'])
    else:
        train_dataset = MT5Dataset(train_examples, tokenizer, config['max_seq_length'], config['max_answer_length'])
        val_dataset = MT5Dataset(val_examples, tokenizer, config['max_seq_length'], config['max_answer_length'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create trainer
    from src.training.zero_shot_trainer import ZeroShotTrainer
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
        tracker.finish()
    
    logger.info(f"Experiment {args.experiment_name} completed")

if __name__ == "__main__":
    patched_main()
PYTHON_SCRIPT

# Run training with minimal settings (reduced batch size for MPS memory)
caffeinate -i python /tmp/train_mt5_minimal.py \
    --model mt5 \
    --data-path "$DATA_PATH" \
    --batch-size 2 \
    --num-epochs 1 \
    --learning-rate 3e-5 \
    2>&1 | tee "$LOG_FILE"

# Cleanup
rm /tmp/train_mt5_minimal.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Quick comparison training completed!"
    echo "📊 Model saved to: models/checkpoints/"
    echo "📝 Log: $LOG_FILE"
    echo ""
    echo "💡 You can now compare mBERT vs mT5 using:"
    echo "   python scripts/compare_models.py"
else
    echo ""
    echo "❌ Training failed. Check: $LOG_FILE"
    exit 1
fi

# Show notification
osascript -e 'display notification "Quick mT5 training done! Ready for comparison." with title "Training Complete"' 2>/dev/null || true

