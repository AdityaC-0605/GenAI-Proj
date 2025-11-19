# Step-by-Step Guide: Cross-Lingual Question Answering System

## 📖 What This Project Is

This is a **research platform** that compares two multilingual AI models (mBERT and mT5) for answering questions across different languages. The key capability is **cross-lingual question answering** - you can ask a question in one language (e.g., English) and get an answer from a document in another language (e.g., Spanish).

### Key Concepts:
- **mBERT**: Extractive model (finds answers directly from text)
- **mT5**: Generative model (generates answers)
- **Zero-shot**: Train on English, test on other languages
- **Few-shot**: Fine-tune with a few examples per language

---

## 🏗️ What's Built

### ✅ Core Components:
1. **Training System**: Train models on SQuAD dataset
2. **Evaluation System**: Test models on multilingual datasets (XQuAD, MLQA, TyDiQA)
3. **API Server**: REST API for making predictions
4. **Dashboard**: Streamlit web interface for visualization
5. **Comparison Tools**: Compare mBERT vs mT5 performance

### ✅ Supported Languages:
- **Question Languages**: English, Spanish, French, German, Chinese, Arabic (6 languages)
- **Context Languages**: All above + Hindi, Japanese, Korean (9 languages)
- **Total Language Pairs**: 54 combinations

### ✅ Datasets:
- **SQuAD 2.0**: English training data (130K+ examples)
- **XQuAD**: 11 languages for evaluation
- **MLQA**: 49 language pairs for evaluation
- **TyDiQA**: 11 typologically diverse languages

---

## 🚀 Step-by-Step Guide to Get Results

### **STEP 1: Environment Setup** (5 minutes)

```bash
# Navigate to project directory
cd /Users/aditya/Downloads/Bert_VS_T5

# Activate virtual environment (if not already activated)
source venv/bin/activate

# Verify installation
python --version  # Should be Python 3.9+
pip list | grep torch  # Should show PyTorch installed
```

**What this does**: Ensures your Python environment is ready.

---

### **STEP 2: Download Data** (10-30 minutes, depending on internet speed)

```bash
# Download SQuAD 2.0 (required for training)
python scripts/download_data.py --dataset squad

# Download XQuAD (optional, for cross-lingual evaluation)
python scripts/download_data.py --dataset xquad

# Verify data is downloaded
ls -lh data/squad/
# Should see: train-v2.0.json and dev-v2.0.json
```

**What this does**: Downloads the datasets needed for training and evaluation.

**Expected output**: 
- `data/squad/train-v2.0.json` (~90MB)
- `data/squad/dev-v2.0.json` (~4MB)

---

### **STEP 3: Quick Training (Recommended for First Time)** (15-30 minutes)

For a **quick comparison** between mBERT and mT5 with minimal training:

```bash
# Train mT5 quickly (optimized for Apple Silicon)
./train_mt5_comparison.sh data/squad/train-v2.0.json
```

**What this does**:
- Trains mT5 on ~1,500 examples (subset of SQuAD)
- Uses 1 epoch with optimized batch size
- Saves model to `models/checkpoints/zero_shot/best_model.pt`

**Expected output**: 
- Training logs in `logs/` directory
- Model checkpoint saved
- Experiment tracking JSON in `experiments/tracking/`

**OR** for mBERT (if you want to train both):

```bash
# Train mBERT (faster, smaller model)
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 16 \
    --num-epochs 1 \
    --output-dir models/mbert
```

---

### **STEP 4: Evaluate Models** (5-10 minutes)

After training, evaluate your model:

```bash
# Evaluate mT5 on SQuAD dev set
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/checkpoints/zero_shot/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad

# Evaluate mBERT on SQuAD dev set
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad
```

**What this does**: Tests the model on the development set and calculates metrics (Exact Match, F1 Score).

**Expected output**:
```
Evaluation Results:
- Exact Match: 0.XX
- F1 Score: 0.XX
- Total Examples: XXXX
Results saved to: experiments/evaluations/...
```

---

### **STEP 5: Cross-Lingual Evaluation** (10-20 minutes)

Test how well your model works across languages:

```bash
# Evaluate on XQuAD (Spanish)
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/checkpoints/zero_shot/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.es.json \
    --dataset-type xquad

# Evaluate on XQuAD (French)
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/checkpoints/zero_shot/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.fr.json \
    --dataset-type xquad
```

**What this does**: Tests zero-shot performance - model trained on English, tested on other languages.

**Expected output**: Performance metrics for each language, showing cross-lingual transfer capability.

---

### **STEP 6: Compare Models** (2-5 minutes)

Compare mBERT vs mT5 performance:

```bash
# Find your experiment results
ls experiments/tracking/

# Compare two experiments
python scripts/compare_models.py \
    --results-a experiments/tracking/zero_shot_mbert_*.json \
    --results-b experiments/tracking/zero_shot_mt5_*.json \
    --model-a-name mBERT \
    --model-b-name mT5
```

**What this does**: Generates statistical comparison showing which model performs better.

**Expected output**: 
- Side-by-side metrics comparison
- Statistical significance tests
- Performance by language pair

---

### **STEP 7: Use the API** (5 minutes)

Start the API server to make predictions:

```bash
# Start API server
./start_api.sh

# OR manually:
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

**In another terminal**, test the API:

```bash
# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "context": "Paris is the capital and most populous city of France.",
    "question_language": "en",
    "context_language": "en",
    "model_name": "mt5"
  }'
```

**Expected output**: JSON response with answer, confidence, and processing time.

---

### **STEP 8: Use the Dashboard** (Optional, 5 minutes)

Launch the interactive web dashboard:

```bash
# Install Streamlit dependencies (if not already installed)
pip install -r streamlit-requirements.txt

# Start dashboard
streamlit run app.py
```

**What this does**: Opens a web interface at `http://localhost:8501` where you can:
- Ask questions interactively
- View model comparisons
- Explore datasets
- Monitor training progress

**Note**: The API server must be running for the dashboard to work.

---

## 📊 Understanding Your Results

### Key Metrics:

1. **Exact Match (EM)**: Percentage of answers that match exactly
   - Range: 0.0 to 1.0
   - Higher is better
   - Example: 0.75 = 75% exact matches

2. **F1 Score**: Token-level overlap between predicted and correct answers
   - Range: 0.0 to 1.0
   - Higher is better
   - Example: 0.82 = 82% F1 score

3. **BLEU/ROUGE**: For generative models (mT5)
   - Measures quality of generated text
   - Higher is better

### What Good Results Look Like:

- **English (same language)**: EM ~0.75-0.85, F1 ~0.80-0.90
- **Cross-lingual (zero-shot)**: EM ~0.60-0.75, F1 ~0.70-0.85
- **Few-shot (with examples)**: EM ~0.70-0.80, F1 ~0.75-0.88

---

## 🔄 Complete Workflow Summary

```
1. Setup Environment          → 5 min
2. Download Data              → 10-30 min
3. Train Model (Quick)        → 15-30 min
4. Evaluate Model             → 5-10 min
5. Cross-Lingual Evaluation   → 10-20 min
6. Compare Models             → 2-5 min
7. Use API                    → 5 min
8. Use Dashboard (Optional)   → 5 min
─────────────────────────────────────
Total Time: ~1-2 hours
```

---

## 🎯 Quick Start (Fastest Path to Results)

If you want to see results **as quickly as possible**:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Download data (if not already done)
python scripts/download_data.py --dataset squad

# 3. Quick train mT5
./train_mt5_comparison.sh data/squad/train-v2.0.json

# 4. Evaluate
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/checkpoints/zero_shot/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad

# 5. Start API and test
./start_api.sh
# Then test with curl or dashboard
```

**Time**: ~30-45 minutes total

---

## 📁 Where to Find Results

- **Model Checkpoints**: `models/checkpoints/` or `models/mbert/`
- **Evaluation Results**: `experiments/evaluations/`
- **Experiment Tracking**: `experiments/tracking/*.json`
- **Training Logs**: `logs/training_*.log`

---

## 🐛 Troubleshooting

### Issue: "Out of Memory" Error
**Solution**: Use the quick training script (already optimized) or reduce batch size:
```bash
python scripts/train_zero_shot.py --model mt5 --batch-size 2 --num-epochs 1
```

### Issue: "API Server Offline"
**Solution**: Make sure API server is running:
```bash
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### Issue: "Model checkpoint not found"
**Solution**: Check if training completed successfully:
```bash
ls -lh models/checkpoints/zero_shot/
ls -lh models/mbert/
```

### Issue: "Dataset not found"
**Solution**: Download the dataset:
```bash
python scripts/download_data.py --dataset squad
python scripts/download_data.py --dataset xquad
```

---

## 📚 Next Steps

1. **Explore Notebooks**: Check out `notebooks/` for detailed analysis
   - `01_data_exploration.ipynb`: Understand the data
   - `02_model_training.ipynb`: Training workflow
   - `03_evaluation_visualization.ipynb`: Results visualization
   - `04_api_usage.ipynb`: API examples

2. **Try Few-Shot Learning**: Fine-tune with few examples
   ```bash
   python scripts/train_few_shot.py --model mt5 --num-shots 10
   ```

3. **Experiment with Different Languages**: Test various language pairs
   ```bash
   python scripts/run_language_pair_analysis.sh
   ```

4. **Read Documentation**: 
   - `README.md`: Full project documentation
   - `PROJECT_ANALYSIS.md`: Technical analysis
   - `QUICK_MT5_COMPARISON.md`: Quick training guide

---

## 💡 Key Takeaways

1. **This project compares mBERT vs mT5** for cross-lingual QA
2. **Zero-shot learning**: Train on English, test on other languages
3. **54 language pairs** supported
4. **Quick training** takes ~30 minutes, full training takes hours
5. **Results are saved** in `experiments/` directory
6. **API and Dashboard** provide easy interfaces for testing

---

## ❓ Common Questions

**Q: Do I need to train both models?**
A: No, you can train just one. But comparing both gives you the full research picture.

**Q: How long does full training take?**
A: Full training (all data, 3 epochs) can take 2-6 hours depending on your hardware.

**Q: Can I use pre-trained models?**
A: The models use pre-trained weights from Hugging Face, but you need to fine-tune them on SQuAD for QA.

**Q: What's the difference between zero-shot and few-shot?**
A: Zero-shot = train on English only. Few-shot = add a few examples from target languages.

---

**Happy Experimenting! 🚀**

