# How to Get Better Results

## 🔴 Problem Identified

Your training logs show **`Loss: nan`** (NaN - Not a Number) throughout training. This means:
- ❌ Training failed - models didn't learn properly
- ❌ Models are essentially random/untrained
- ❌ That's why evaluation scores are 0.03% (essentially random guessing)

## ✅ Solution: Retrain Properly

### Step 1: Retrain mBERT (2-4 hours)

```bash
cd /Users/aditya/Downloads/Bert_VS_T5
source venv/bin/activate

# Retrain mBERT with proper settings
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 16 \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --output-dir models/mbert_retrained
```

**What to watch for:**
- ✅ Loss should be a **number** (e.g., 2.5, 1.8, 1.2) - NOT `nan`
- ✅ Loss should **decrease** over time
- ✅ Training should complete without errors

**Expected time:** 2-4 hours on Apple Silicon

---

### Step 2: Retrain mT5 (3-6 hours)

```bash
# Retrain mT5 with proper settings
python scripts/train_zero_shot.py \
    --model mt5 \
    --data-path data/squad/train-v2.0.json \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --output-dir models/mt5_retrained
```

**What to watch for:**
- ✅ Loss should be a **number** (e.g., 3.2, 2.1, 1.5) - NOT `nan`
- ✅ Loss should **decrease** over time
- ⚠️ mT5 is larger, so training is slower

**Expected time:** 3-6 hours on Apple Silicon

---

### Step 3: Monitor Training

**In a separate terminal**, watch the training progress:

```bash
# Watch training logs in real-time
tail -f logs/training_*.log

# Or check process
ps aux | grep train_zero_shot
```

**Good signs:**
- Loss values are numbers (not `nan`)
- Loss decreases: `Loss: 2.5` → `Loss: 1.8` → `Loss: 1.2`
- No error messages

**Bad signs:**
- `Loss: nan` (training failed)
- Error messages
- Process crashes

---

### Step 4: Evaluate Retrained Models

After training completes:

```bash
# Evaluate retrained mBERT
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert_retrained/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad

# Evaluate retrained mT5
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/mt5_retrained/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad
```

**Expected results after proper training:**
- **mBERT**: EM ~0.70-0.80, F1 ~0.80-0.90
- **mT5**: EM ~0.65-0.75, F1 ~0.75-0.85

---

## 🔧 Troubleshooting NaN Loss

If you still get `Loss: nan`, try:

### Option A: Use CPU Instead of MPS

```bash
# Force CPU mode (slower but more stable)
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 8 \
    --num-epochs 3 \
    --output-dir models/mbert_retrained
```

### Option B: Reduce Learning Rate

```bash
# Try lower learning rate
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 16 \
    --num-epochs 3 \
    --learning-rate 1e-5 \
    --output-dir models/mbert_retrained
```

### Option C: Use Gradient Clipping

```bash
# Add gradient clipping (already in config, but verify)
# Check configs/training/zero_shot.yaml has:
# max_grad_norm: 1.0
```

---

## 📊 Quick Training Script

I'll create a script that handles common issues:

```bash
# Save this as retrain_models.sh
#!/bin/bash
set -e

cd /Users/aditya/Downloads/Bert_VS_T5
source venv/bin/activate

echo "=== Retraining mBERT ==="
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 16 \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --output-dir models/mbert_retrained

echo "=== Retraining mT5 ==="
python scripts/train_zero_shot.py \
    --model mt5 \
    --data-path data/squad/train-v2.0.json \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 3e-5 \
    --output-dir models/mt5_retrained

echo "=== Training Complete! ==="
```

---

## 🎯 Action Plan

**Right Now:**

1. **Start mBERT retraining** (run the command in Step 1)
2. **Monitor for NaN loss** - if you see `nan`, stop and try troubleshooting
3. **Let it run** - 2-4 hours for mBERT
4. **Then retrain mT5** - 3-6 hours

**After Training:**

1. Evaluate both models
2. Compare results
3. You should see **much better** scores (70-90% instead of 0.03%)

---

## ⚠️ Important Notes

1. **Prevent sleep** during training:
   ```bash
   caffeinate -i python scripts/train_zero_shot.py ...
   ```

2. **Check disk space** - models need ~2-7GB each:
   ```bash
   df -h .
   ```

3. **Training takes time** - be patient, don't interrupt

4. **Watch the logs** - make sure loss is decreasing, not `nan`

---

## 💡 Why This Will Work

- **More epochs** (3 instead of 1) = better learning
- **Proper loss values** (not NaN) = actual training
- **Better hyperparameters** = optimized learning
- **Full dataset** = more training data

**You should see 100-1000x improvement in scores!** 🚀

