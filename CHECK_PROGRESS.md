# How to Check Training Progress When You Wake Up

## Quick Check Commands

### 1. Check if training is still running:
```bash
ps aux | grep train_zero_shot | grep -v grep
```

If you see a process, training is still running.

### 2. Check the latest log file:
```bash
# Find the latest log
ls -lt logs/overnight_training/ | head -5

# View the latest log
tail -50 logs/overnight_training/training_*.log | tail -50
```

### 3. Check if models were created:
```bash
# Check mBERT
ls -lh models/mbert_retrained/best_model.pt 2>/dev/null && echo "✅ mBERT trained" || echo "❌ mBERT not ready"

# Check mT5
ls -lh models/mt5_retrained/best_model.pt 2>/dev/null && echo "✅ mT5 trained" || echo "❌ mT5 not ready"
```

### 4. Check for errors (NaN loss):
```bash
grep -i "nan\|error\|failed" logs/overnight_training/training_*.log | tail -20
```

### 5. See training summary:
```bash
# Show final status
tail -30 logs/overnight_training/training_*.log | grep -E "Complete|Finished|failed|ERROR"
```

## What to Look For

### ✅ Good Signs:
- "Training completed successfully"
- Loss values are numbers (e.g., "Loss: 1.2", "Loss: 0.8")
- Loss decreases over time
- Model files exist: `models/mbert_retrained/best_model.pt`

### ❌ Bad Signs:
- "Loss: nan" (training failed)
- "ERROR" or "failed" messages
- Process crashed (no process running)
- Model files don't exist

## If Training Completed Successfully

Run evaluations:

```bash
source venv/bin/activate

# Evaluate mBERT
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert_retrained/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad

# Evaluate mT5
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/mt5_retrained/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad
```

**Expected results:** EM ~70-80%, F1 ~80-90% (much better than 0.03%!)

## If Training Failed

1. Check the log file for errors
2. Look for NaN loss issues
3. You may need to retrain with different settings
4. Check disk space: `df -h .`

