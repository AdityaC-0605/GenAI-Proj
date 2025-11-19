#!/bin/bash
# Overnight Training Script - Trains both mBERT and mT5 sequentially
# This will run for 5-10 hours total

set -e  # Exit on error

cd /Users/aditya/Downloads/Bert_VS_T5
source venv/bin/activate

# Create log directory
mkdir -p logs/overnight_training
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/overnight_training/training_${TIMESTAMP}.log"

echo "==========================================" | tee -a "$LOG_FILE"
echo "Overnight Training Started: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "This script will:" | tee -a "$LOG_FILE"
echo "1. Train mBERT (2-4 hours)" | tee -a "$LOG_FILE"
echo "2. Train mT5 (3-6 hours)" | tee -a "$LOG_FILE"
echo "3. Total time: 5-10 hours" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "You can check progress with: tail -f $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to check for NaN in logs
check_for_nan() {
    local model=$1
    if tail -20 "$LOG_FILE" | grep -q "Loss: nan"; then
        echo "⚠️  WARNING: NaN loss detected for $model!" | tee -a "$LOG_FILE"
        echo "Training may have failed. Check logs when you wake up." | tee -a "$LOG_FILE"
        return 1
    fi
    return 0
}

# Step 1: Train mBERT
echo "==========================================" | tee -a "$LOG_FILE"
echo "Step 1/2: Training mBERT" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 16 \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --output-dir models/mbert_retrained \
    2>&1 | tee -a "$LOG_FILE"

MBERT_EXIT_CODE=${PIPESTATUS[0]}

if [ $MBERT_EXIT_CODE -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "✅ mBERT training completed successfully!" | tee -a "$LOG_FILE"
    echo "Finished: $(date)" | tee -a "$LOG_FILE"
    check_for_nan "mBERT"
else
    echo "" | tee -a "$LOG_FILE"
    echo "❌ mBERT training failed with exit code $MBERT_EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check logs for details" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Waiting 30 seconds before starting mT5..." | tee -a "$LOG_FILE"
sleep 30

# Step 2: Train mT5
echo "==========================================" | tee -a "$LOG_FILE"
echo "Step 2/2: Training mT5" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python scripts/train_zero_shot.py \
    --model mt5 \
    --data-path data/squad/train-v2.0.json \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --output-dir models/mt5_retrained \
    2>&1 | tee -a "$LOG_FILE"

MT5_EXIT_CODE=${PIPESTATUS[0]}

if [ $MT5_EXIT_CODE -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "✅ mT5 training completed successfully!" | tee -a "$LOG_FILE"
    echo "Finished: $(date)" | tee -a "$LOG_FILE"
    check_for_nan "mT5"
else
    echo "" | tee -a "$LOG_FILE"
    echo "❌ mT5 training failed with exit code $MT5_EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check logs for details" | tee -a "$LOG_FILE"
fi

# Final summary
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Overnight Training Complete!" | tee -a "$LOG_FILE"
echo "Ended: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $MBERT_EXIT_CODE -eq 0 ] && [ $MT5_EXIT_CODE -eq 0 ]; then
    echo "✅ Both models trained successfully!" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Next steps when you wake up:" | tee -a "$LOG_FILE"
    echo "1. Evaluate mBERT:" | tee -a "$LOG_FILE"
    echo "   python scripts/evaluate.py --model mbert --checkpoint models/mbert_retrained/best_model.pt --data-path data/squad/dev-v2.0.json --dataset-type squad" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "2. Evaluate mT5:" | tee -a "$LOG_FILE"
    echo "   python scripts/evaluate.py --model mt5 --checkpoint models/mt5_retrained/best_model.pt --data-path data/squad/dev-v2.0.json --dataset-type squad" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "3. Compare results - you should see much better scores!" | tee -a "$LOG_FILE"
else
    echo "⚠️  Some training failed. Check the log file for details:" | tee -a "$LOG_FILE"
    echo "   $LOG_FILE" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Log file saved to: $LOG_FILE" | tee -a "$LOG_FILE"

