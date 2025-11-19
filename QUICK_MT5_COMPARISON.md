# Quick mT5 Training for Comparison

This guide helps you quickly train mT5 with minimal data to compare with your already-trained mBERT model.

## Quick Start

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run quick training
./train_mt5_comparison.sh data/squad/train-v2.0.json
```

## Training Configuration

The quick training uses:
- **Training examples**: ~1,500 (minimal subset from full dataset)
- **Validation examples**: ~200 (subset for faster evaluation)
- **Batch size**: 2 (optimized for MPS memory)
- **Gradient accumulation**: 8 (effective batch size: 16)
- **Epochs**: 1
- **Learning rate**: 3e-5
- **Expected time**: 15-30 minutes (depending on hardware)
- **Device**: MPS (Mac GPU) with automatic memory optimization

## What Gets Saved

- Model checkpoint: `models/checkpoints/zero_shot/`
- Training logs: `logs/training_mt5_comparison_*.log`
- Experiment tracking: `experiments/tracking/`

## After Training

Once training completes, you can:

1. **Evaluate the model**:
   ```bash
   python scripts/evaluate.py \
       --model mt5 \
       --checkpoint models/checkpoints/zero_shot/best_model.pt \
       --data-path data/squad/dev-v2.0.json
   ```

2. **Compare with mBERT**:
   ```bash
   python scripts/compare_models.py \
       --results-a experiments/tracking/zero_shot_mbert_*.json \
       --results-b experiments/tracking/zero_shot_mt5_*.json \
       --model-a-name mBERT \
       --model-b-name mT5
   ```

## Notes

- This is a **minimal training** for quick comparison only
- For production use, train with full dataset and more epochs
- The model will be less accurate than full training but sufficient for comparison
- Both models should be evaluated on the same test set for fair comparison

