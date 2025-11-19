# Cross-Lingual Question Answering System

A research platform for comparing multilingual BERT (mBERT) and multilingual T5 (mT5) models on cross-lingual question answering tasks. The system supports zero-shot and few-shot learning across multiple language pairs.

## 📚 Getting Started Guides

**New to this project?** Start here:

- **[STEP_BY_STEP_GUIDE.md](STEP_BY_STEP_GUIDE.md)** - Complete step-by-step instructions to get results (recommended for beginners)
- **[PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md)** - Visual workflow diagrams and system architecture explanation
- **[PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md)** - Technical analysis of what's implemented

## Features

- **Dual Model Architecture**: Compare mBERT (extractive) and mT5 (generative) approaches
- **Cross-Lingual Support**: 36+ language pair combinations
- **Zero-Shot Learning**: Train on English, test on other languages
- **Few-Shot Learning**: Fine-tune with 1, 5, 10, or 50 examples per language pair
- **Apple Silicon Optimization**: MPS backend support with automatic CPU fallback
- **Comprehensive Evaluation**: Exact Match, F1, BLEU, ROUGE metrics with statistical analysis
- **REST API**: FastAPI-based inference endpoint
- **Streamlit Dashboard**: Interactive visualization and model comparison

## Quick Start

### 1. Setup

```bash
# Clone and navigate to project
cd Bert_VS_T5

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Download Data

```bash
# Download SQuAD 2.0 for training
python scripts/download_data.py --dataset squad

# Download XQuAD for cross-lingual evaluation (optional)
python scripts/download_data.py --dataset xquad
```

### 3. Quick Training for Comparison

**For quick comparison between mBERT and mT5** (minimal training, ~15-30 minutes):

```bash
# Train mT5 with minimal data for quick comparison
./train_mt5_comparison.sh data/squad/train-v2.0.json
```

This will:
- Train mT5 on ~1,500 examples (minimal subset)
- Use 1 epoch with optimized batch size for MPS memory
- Complete in 15-30 minutes
- Save model to `models/checkpoints/zero_shot/`

See [QUICK_MT5_COMPARISON.md](QUICK_MT5_COMPARISON.md) for detailed information.

### 4. Full Training

**For production-quality models** (full dataset, multiple epochs):

```bash
# Train mBERT
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path data/squad/train-v2.0.json \
    --batch-size 16 \
    --num-epochs 3

# Train mT5
python scripts/train_zero_shot.py \
    --model mt5 \
    --data-path data/squad/train-v2.0.json \
    --batch-size 8 \
    --num-epochs 3
```

### 5. Evaluate Models

```bash
# Evaluate mBERT on SQuAD dev set
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert/best_model.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad

# Evaluate mT5 on SQuAD dev set
# Note: Use checkpoint_epoch_1.pt if best_model.pt doesn't exist
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint models/checkpoints/checkpoint_epoch_1.pt \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad

# Evaluate on cross-lingual datasets (XQuAD, MLQA, TyDiQA)
python scripts/evaluate.py \
    --model mbert \
    --checkpoint models/mbert/best_model.pt \
    --data-path data/xquad/xquad-master/xquad.en.json \
    --dataset-type xquad
```

### 6. Compare Models

```bash
python scripts/compare_models.py \
    --results-a experiments/tracking/zero_shot_mbert_*.json \
    --results-b experiments/tracking/zero_shot_mt5_*.json \
    --model-a-name mBERT \
    --model-b-name mT5
```

## Project Structure

```
Bert_VS_T5/
├── src/
│   ├── data/              # Data loaders and preprocessors
│   ├── models/            # mBERT and mT5 implementations
│   ├── training/          # Training pipelines
│   ├── evaluation/        # Evaluation metrics and comparison
│   ├── inference/         # Inference engine
│   ├── api/               # REST API server
│   └── utils/             # Utility functions
├── configs/               # Configuration files
│   ├── model/            # Model configs
│   ├── training/         # Training configs
│   └── dataset/          # Dataset configs
├── scripts/              # Training and evaluation scripts
├── data/                 # Dataset storage
├── models/               # Model checkpoints
├── experiments/          # Experiment results
├── logs/                 # Training logs
└── notebooks/            # Jupyter notebooks for exploration
```

## Supported Datasets

- **SQuAD 2.0**: English question answering with unanswerable questions
- **XQuAD**: Cross-lingual QA in 11 languages
- **MLQA**: Multilingual QA with 49 language pairs
- **TyDi QA**: Typologically diverse QA in 11 languages

## API Usage

### Start API Server

```bash
./start_api.sh

# Or manually
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

API will be available at:
- **API endpoint**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

### Make Predictions

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "question": "What is the capital of France?",
        "context": "Paris is the capital and most populous city of France.",
        "question_language": "en",
        "context_language": "en",
        "model_id": "mbert_zero_shot"
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Streamlit Dashboard

Launch the interactive dashboard:

```bash
# Install Streamlit dependencies
pip install -r streamlit-requirements.txt

# Start dashboard

# or
streamlit run app.py
```

The dashboard provides:
- 🏠 **Home**: System overview
- ❓ **Ask Questions**: Interactive QA interface
- 📊 **Model Comparison**: Performance analysis
- 📈 **Training Monitor**: Real-time training progress
- 📁 **Dataset Explorer**: Data visualization

**Note**: The dashboard requires the API server to be running. See [RUN_DASHBOARD.md](RUN_DASHBOARD.md) for details.

## Evaluation Metrics

- **Exact Match (EM)**: Binary match after normalization
- **F1 Score**: Token-level F1 score
- **BLEU**: For generative models (mT5)
- **ROUGE**: For generative models (mT5)
- **Statistical Tests**: Paired t-tests, bootstrap confidence intervals

## Performance Optimization

### Apple Silicon (M1/M2/M3)

The system automatically detects and uses the MPS backend. For memory-constrained training:

```bash
# Use the quick comparison script (optimized for MPS)
./train_mt5_comparison.sh data/squad/train-v2.0.json
```

The script automatically:
- Reduces batch size to 2 (with gradient accumulation)
- Sets MPS memory limits appropriately
- Uses mixed precision training

### Memory Management

If you encounter out-of-memory errors:

```bash
# Reduce batch size and increase gradient accumulation
python scripts/train_zero_shot.py \
    --model mt5 \
    --data-path data/squad/train-v2.0.json \
    --batch-size 2 \
    --num-epochs 1
```

## Troubleshooting

### MPS Backend Issues

If you encounter MPS-related errors:

```bash
# The quick training script handles this automatically
# For manual training, you can force CPU mode:
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Out of Memory

The `train_mt5_comparison.sh` script is optimized for MPS memory constraints. If issues persist:

1. Close other applications using GPU memory
2. Reduce batch size further (edit the script)
3. Use CPU mode (slower but more stable)

## Configuration

Training parameters can be adjusted in:
- `configs/training/zero_shot.yaml` - Zero-shot training config
- `configs/training/few_shot.yaml` - Few-shot training config
- `configs/model/mbert.yaml` - mBERT model config
- `configs/model/mt5.yaml` - mT5 model config

## Citation

If you use this system in your research, please cite:

```bibtex
@software{cross_lingual_qa,
  title={Cross-Lingual Question Answering System},
  author={Research Team},
  year={2024},
  url={https://github.com/example/cross-lingual-qa}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Hugging Face Transformers library
- PyTorch team for MPS backend support
- SQuAD, XQuAD, MLQA, and TyDi QA dataset creators
