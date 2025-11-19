#!/bin/bash
# Script to retrain models for better results

set -e  # Exit on error

cd /Users/aditya/Downloads/Bert_VS_T5
source venv/bin/activate

echo "=========================================="
echo "Retraining Models for Better Results"
echo "=========================================="
echo ""
echo "Problem: Previous training had NaN losses"
echo "Solution: Retrain with proper settings"
echo ""
echo "This will take several hours. Make sure:"
echo "1. Your system won't sleep (use caffeinate)"
echo "2. You have enough disk space (~10GB)"
echo "3. You can leave it running"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

echo ""
echo "=== Step 1: Retraining mBERT ==="
echo "This will take 2-4 hours..."
echo ""

# Retrain mBERT
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 16 \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --output-dir models/mbert_retrained

echo ""
echo "✅ mBERT training complete!"
echo ""
echo "=== Step 2: Retraining mT5 ==="
echo "This will take 3-6 hours..."
echo ""

# Retrain mT5
python scripts/train_zero_shot.py \
    --model mt5 \
    --data-path data/squad/train-v2.0.json \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --output-dir models/mt5_retrained

echo ""
echo "✅ mT5 training complete!"
echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Evaluate the retrained models:"
echo "   python scripts/evaluate.py --model mbert --checkpoint models/mbert_retrained/best_model.pt --data-path data/squad/dev-v2.0.json --dataset-type squad"
echo ""
echo "2. Compare results - you should see much better scores!"
echo ""

