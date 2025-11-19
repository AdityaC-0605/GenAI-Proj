# Next Steps After Evaluation

## ✅ What You've Completed

1. ✅ **mBERT Evaluation**: Completed (1 hour 2 minutes)
   - Results: EM=0.0003, F1=0.0013
   - Saved to: `experiments/evaluations/mbert_squad_20251117_230415.json`

2. ✅ **mT5 Evaluation**: Completed (31 minutes)
   - Results: EM=0.0003, F1=0.0003
   - Saved to: `experiments/evaluations/mt5_squad_20251118_003933.json`

## ⚠️ Note on Results

The evaluation scores are very low (0.03% EM, 0.1-0.3% F1). This could indicate:
- Models need better training (more epochs, better hyperparameters)
- Checkpoint loading issues
- Evaluation metric calculation issues

**However**, you can still proceed with the next steps to:
- Compare the models
- Test cross-lingual performance
- Use the API/dashboard
- Investigate the results

---

## 🎯 Recommended Next Steps

### Step 1: Compare Models (2 minutes)

Compare mBERT vs mT5 performance:

```bash
python scripts/compare_models.py \
    --results-a experiments/evaluations/mbert_squad_20251117_230415.json \
    --results-b experiments/evaluations/mt5_squad_20251118_003933.json \
    --model-a-name mBERT \
    --model-b-name mT5
```

**What this does**: Shows side-by-side comparison and statistical analysis.

---

### Step 2: Cross-Lingual Evaluation (Optional, 30-60 min per language)

Test how models perform on other languages:

```bash
# Evaluate mBERT on Spanish (XQuAD)
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.es.json \
    --dataset-type xquad

# Evaluate mBERT on French
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.fr.json \
    --dataset-type xquad
```

**What this tests**: Zero-shot cross-lingual transfer (train on English, test on other languages).

---

### Step 3: Use the API (5 minutes)

Start the API server and test predictions:

```bash
# Start API server
./start_api.sh

# In another terminal, test it
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "context": "Paris is the capital and most populous city of France.",
    "question_language": "en",
    "context_language": "en",
    "model_name": "mbert"
  }'
```

**What this does**: Lets you interactively test the models with your own questions.

---

### Step 4: Launch Dashboard (5 minutes)

Visualize results and interact with models:

```bash
# Install Streamlit dependencies (if not already)
pip install -r streamlit-requirements.txt

# Start dashboard
streamlit run app.py
```

**What this does**: Opens web interface at http://localhost:8501 with:
- Interactive QA interface
- Model comparison visualizations
- Dataset explorer

**Note**: API server must be running for dashboard to work.

---

### Step 5: Investigate Low Scores (Optional)

If you want to understand why scores are low:

1. **Check training logs**:
   ```bash
   tail -50 logs/training_*.log
   ```

2. **Verify model checkpoints**:
   ```bash
   ls -lh models/mbert/best_model.pt
   ls -lh models/checkpoints/checkpoint_epoch_1.pt
   ```

3. **Test on a small sample**:
   ```bash
   # Create a test script to check a few examples manually
   python -c "
   from src.models.mbert_wrapper import MBERTModelWrapper
   model = MBERTModelWrapper()
   model.load_checkpoint('models/mbert/best_model.pt')
   result = model.predict(
       question='What is the capital of France?',
       context='Paris is the capital of France.',
       question_lang='en',
       context_lang='en'
   )
   print(f'Answer: {result.answer_text}')
   print(f'Confidence: {result.confidence}')
   "
   ```

---

### Step 6: Retrain Models (If Needed, 1-4 hours)

If you want better performance, retrain with more epochs:

```bash
# Retrain mBERT with 3 epochs
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 16 \
    --num-epochs 3 \
    --output-dir models/mbert

# Retrain mT5 with 3 epochs
python scripts/train_zero_shot.py \
    --model mt5 \
    --data-path data/squad/train-v2.0.json \
    --batch-size 8 \
    --num-epochs 3 \
    --output-dir models/checkpoints/zero_shot
```

---

## 📊 Quick Comparison Summary

| Model | EM Score | F1 Score | Evaluation Time |
|-------|----------|----------|-----------------|
| **mBERT** | 0.0003 (0.03%) | 0.0013 (0.13%) | 1h 2m |
| **mT5** | 0.0003 (0.03%) | 0.0003 (0.03%) | 31m |

**Observation**: Both models show similar very low performance. mT5 was faster to evaluate.

---

## 🎯 What I Recommend Doing Next

**Priority Order:**

1. **Compare models** (Step 1) - Quick, shows you the comparison
2. **Test API** (Step 3) - See if models work better interactively
3. **Launch Dashboard** (Step 4) - Visual exploration
4. **Cross-lingual evaluation** (Step 2) - If you want to test multilingual capability
5. **Investigate/Retrain** (Steps 5-6) - If you need better performance

---

## 💡 Quick Commands Reference

```bash
# Compare models
python scripts/compare_models.py \
    --results-a experiments/evaluations/mbert_squad_20251117_230415.json \
    --results-b experiments/evaluations/mt5_squad_20251118_003933.json \
    --model-a-name mBERT \
    --model-b-name mT5

# Start API
./start_api.sh

# Start Dashboard
streamlit run app.py

# Cross-lingual test (Spanish)
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.es.json \
    --dataset-type xquad
```

---

**You're making great progress!** 🚀

