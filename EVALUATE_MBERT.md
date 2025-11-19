# How to Test mBERT Training Results

## ✅ Training Status

Your mBERT training has **completed successfully**! 

**Checkpoints found:**
- `models/mbert_retrained/best_model.pt` (679MB) - **Best model**
- `checkpoint_epoch_1.pt` (2.0GB)
- `checkpoint_epoch_2.pt` (2.0GB)  
- `checkpoint_epoch_3.pt` (2.0GB)

All 3 epochs completed! 🎉

---

## 🧪 Evaluation Tests

### Test 1: SQuAD Dev Set (English - Same Language)

**Purpose**: Test performance on English (same language as training)

**Command**:
```bash
cd /Users/aditya/Downloads/Bert_VS_T5
source venv/bin/activate

python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert_retrained/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad
```

**Expected Results**:
- EM: ~70-85%
- F1: ~80-90%
- Time: ~1-2 hours (11,873 examples)

**What to look for**:
- ✅ Much better than previous 0.03%!
- ✅ Loss decreased properly during training
- ✅ Model learned from data

---

### Test 2: Cross-Lingual Evaluation (Spanish)

**Purpose**: Test zero-shot cross-lingual transfer

**Command**:
```bash
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert_retrained/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.es.json \
    --dataset-type xquad
```

**Expected Results**:
- EM: ~60-75% (lower than English, but good for zero-shot)
- F1: ~70-85%
- Shows cross-lingual transfer capability

---

### Test 3: Multiple Languages (Quick Comparison)

**Purpose**: Compare performance across languages

**Commands**:
```bash
# French
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert_retrained/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.fr.json \
    --dataset-type xquad

# German
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert_retrained/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.de.json \
    --dataset-type xquad

# Chinese
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert_retrained/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.zh.json \
    --dataset-type xquad
```

---

### Test 4: Compare with Previous Results

**Purpose**: See improvement from retraining

**Command**:
```bash
python scripts/compare_models.py \
    --results-a experiments/evaluations/mbert_squad_*_retrained*.json \
    --results-b experiments/evaluations/mbert_squad_20251117_230415.json \
    --model-a-name "mBERT Retrained" \
    --model-b-name "mBERT Original"
```

**Expected**: Huge improvement (0.03% → 70-85%)

---

## 🚀 Quick Start

**Easiest way to test**:

```bash
cd /Users/aditya/Downloads/Bert_VS_T5
source venv/bin/activate
./test_mbert_results.sh
```

This will:
1. Find the best checkpoint
2. Evaluate on SQuAD dev set
3. Show you the results

---

## 📊 Understanding Results

### Good Results Look Like:

**SQuAD (English-English)**:
- EM: 0.70-0.85 (70-85%)
- F1: 0.80-0.90 (80-90%)

**XQuAD (Cross-Lingual)**:
- EM: 0.60-0.75 (60-75%)
- F1: 0.70-0.85 (70-85%)

### Results File Location:

Results are saved to:
```
experiments/evaluations/mbert_squad_TIMESTAMP.json
```

The JSON file contains:
- Exact Match score
- F1 score
- Number of examples
- Language pair breakdown
- Timestamp

---

## 🔍 Quick Test (Sample Examples)

If you want to test on just a few examples first:

```python
from src.models.mbert_wrapper import MBERTModelWrapper

# Load model
model = MBERTModelWrapper()
model.load_checkpoint('models/mbert_retrained/best_model.pt')

# Test example
result = model.predict(
    question="What is the capital of France?",
    context="Paris is the capital and most populous city of France. With a population of more than 2 million people, it is the largest city in France.",
    question_lang="en",
    context_lang="en"
)

print(f"Answer: {result.answer_text}")
print(f"Confidence: {result.confidence:.2f}")
```

**Expected output**:
```
Answer: Paris
Confidence: 0.95
```

---

## ⚠️ Troubleshooting

### Issue: "Checkpoint not found"

**Solution**: Check which checkpoint exists:
```bash
ls -lh models/mbert_retrained/
ls -lh models/mbert/
```

Use the path that exists.

### Issue: "Out of memory"

**Solution**: Evaluation processes one example at a time, so memory shouldn't be an issue. If it is:
- Close other applications
- Use CPU mode (slower but uses less memory)

### Issue: "Evaluation taking too long"

**Solution**: 
- SQuAD dev set has 11,873 examples
- Takes ~1-2 hours
- This is normal! Let it run.

---

## 📈 Next Steps After Evaluation

1. **Check Results**: Look at the JSON file in `experiments/evaluations/`
2. **Compare**: Use `compare_models.py` to see improvement
3. **Cross-Lingual**: Test on other languages
4. **Update Paper**: Add results to your research paper!

---

## 💡 Pro Tips

1. **Start with SQuAD**: Test on English first to verify training worked
2. **Then Cross-Lingual**: Test on other languages to see transfer
3. **Save Results**: Keep evaluation JSON files for comparison
4. **Document**: Note which checkpoint you used in your paper

---

**Ready to test? Run:**
```bash
./test_mbert_results.sh
```

Or manually:
```bash
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert_retrained/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad
```

Good luck! 🚀

