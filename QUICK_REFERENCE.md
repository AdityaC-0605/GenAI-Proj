# Quick Reference Card

## 🚀 Most Common Commands

### Setup & Data
```bash
# Activate environment
source venv/bin/activate

# Download SQuAD (required)
python scripts/download_data.py --dataset squad

# Download XQuAD (for evaluation)
python scripts/download_data.py --dataset xquad
```

### Training
```bash
# Quick training (30 min) - RECOMMENDED FOR FIRST TIME
./train_mt5_comparison.sh data/squad/train-v2.0.json

# Full mBERT training
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 16 \
    --num-epochs 3

# Full mT5 training
python scripts/train_zero_shot.py \
    --model mt5 \
    --data-path data/squad/train-v2.0.json \
    --batch-size 8 \
    --num-epochs 3
```

### Evaluation
```bash
# Evaluate mT5 on SQuAD
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/checkpoints/zero_shot/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad

# Evaluate mBERT on SQuAD
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad

# Cross-lingual evaluation (Spanish)
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/checkpoints/zero_shot/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.es.json \
    --dataset-type xquad
```

### API & Dashboard
```bash
# Start API server
./start_api.sh
# OR
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Start Dashboard
streamlit run app.py
```

### Model Comparison
```bash
# Compare two models
python scripts/compare_models.py \
    --results-a experiments/tracking/zero_shot_mbert_*.json \
    --results-b experiments/tracking/zero_shot_mt5_*.json \
    --model-a-name mBERT \
    --model-b-name mT5
```

---

## 📁 Important File Locations

| What | Where |
|------|-------|
| **Trained Models** | `models/checkpoints/zero_shot/` or `models/mbert/` |
| **Evaluation Results** | `experiments/evaluations/` |
| **Experiment Tracking** | `experiments/tracking/*.json` |
| **Training Logs** | `logs/training_*.log` |
| **Data** | `data/squad/`, `data/xquad/`, etc. |

---

## 🎯 Quick Paths

### Fastest Way to Results (30-45 min)
```bash
source venv/bin/activate
python scripts/download_data.py --dataset squad
./train_mt5_comparison.sh data/squad/train-v2.0.json
python scripts/evaluate.py --model mt5 \
    --checkpoint models/checkpoints/zero_shot/best_model.pt \
    --data-path data/squad/dev-v2.0.json --dataset-type squad
```

### Test API (5 min)
```bash
./start_api.sh
# Then test:
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?", "context": "Paris is the capital of France.", "question_language": "en", "context_language": "en", "model_name": "mt5"}'
```

### View Dashboard (5 min)
```bash
pip install -r streamlit-requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of Memory | Use `train_mt5_comparison.sh` or reduce `--batch-size 2` |
| API Offline | Run `./start_api.sh` first |
| Model Not Found | Check `models/checkpoints/` or `models/mbert/` |
| Dataset Missing | Run `python scripts/download_data.py --dataset squad` |

---

## 📊 Understanding Results

- **Exact Match (EM)**: 0.0-1.0, higher is better (75% = good)
- **F1 Score**: 0.0-1.0, higher is better (80% = good)
- **English Performance**: EM ~0.75-0.85, F1 ~0.80-0.90
- **Cross-Lingual**: EM ~0.60-0.75, F1 ~0.70-0.85

---

## 📚 Full Documentation

- **Step-by-Step Guide**: [STEP_BY_STEP_GUIDE.md](STEP_BY_STEP_GUIDE.md)
- **Workflow Diagrams**: [PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md)
- **Main README**: [README.md](README.md)

---

**Tip**: Bookmark this page for quick access to common commands! 🚀

