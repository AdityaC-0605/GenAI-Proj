# Project Workflow: Understanding the System Architecture

## 🎯 Project Overview

This project is a **Cross-Lingual Question Answering (CLQA) Research Platform** that:
- Compares two AI models: **mBERT** (extractive) vs **mT5** (generative)
- Supports **zero-shot learning** (train on English, test on other languages)
- Supports **few-shot learning** (fine-tune with few examples)
- Works with **54 language pairs** (6 question languages × 9 context languages)

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CROSS-LINGUAL QA SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         DATA LAYER                      │
        │  ┌──────────┐  ┌──────────┐           │
        │  │  SQuAD   │  │  XQuAD   │  MLQA     │
        │  │  (Train) │  │  (Eval)  │  TyDiQA   │
        │  └──────────┘  └──────────┘           │
        │         │              │               │
        │         └──────┬───────┘               │
        │                ▼                       │
        │      Data Loaders & Preprocessors      │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         MODEL LAYER                      │
        │  ┌──────────────┐  ┌──────────────┐   │
        │  │    mBERT     │  │     mT5      │   │
        │  │ (Extractive) │  │ (Generative) │   │
        │  │  110M params │  │  580M params │   │
        │  └──────────────┘  └──────────────┘   │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      TRAINING LAYER                      │
        │  ┌──────────────┐  ┌──────────────┐   │
        │  │ Zero-Shot    │  │  Few-Shot    │   │
        │  │ (English)    │  │ (k examples) │   │
        │  └──────────────┘  └──────────────┘   │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │     EVALUATION LAYER                     │
        │  ┌──────────┐  ┌──────────┐            │
        │  │ Metrics  │  │ Compare  │            │
        │  │ (EM/F1)  │  │ Models   │            │
        │  └──────────┘  └──────────┘            │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      INFERENCE LAYER                    │
        │  ┌──────────┐  ┌──────────┐            │
        │  │   API    │  │Dashboard │            │
        │  │ (FastAPI)│  │(Streamlit)│          │
        │  └──────────┘  └──────────┘            │
        └─────────────────────────────────────────┘
```

---

## 🔄 Complete Workflow Diagram

```
START
  │
  ├─► [1] SETUP ENVIRONMENT
  │     ├─► Activate venv
  │     ├─► Install dependencies
  │     └─► Verify installation
  │
  ├─► [2] DOWNLOAD DATA
  │     ├─► SQuAD (training)
  │     ├─► XQuAD (evaluation)
  │     └─► MLQA, TyDiQA (optional)
  │
  ├─► [3] TRAIN MODEL
  │     │
  │     ├─► ZERO-SHOT TRAINING
  │     │     ├─► Load SQuAD (English)
  │     │     ├─► Train mBERT or mT5
  │     │     ├─► Save checkpoint
  │     │     └─► Track experiment
  │     │
  │     └─► FEW-SHOT TRAINING (optional)
  │           ├─► Load zero-shot checkpoint
  │           ├─► Sample k examples per language
  │           ├─► Fine-tune
  │           └─► Save checkpoint
  │
  ├─► [4] EVALUATE MODEL
  │     ├─► Load checkpoint
  │     ├─► Test on dev set
  │     ├─► Calculate metrics (EM, F1, BLEU, ROUGE)
  │     └─► Save results
  │
  ├─► [5] CROSS-LINGUAL EVALUATION
  │     ├─► Test on XQuAD (multiple languages)
  │     ├─► Test on MLQA (language pairs)
  │     └─► Analyze performance by language
  │
  ├─► [6] COMPARE MODELS
  │     ├─► Load mBERT results
  │     ├─► Load mT5 results
  │     ├─► Statistical analysis
  │     └─► Generate comparison report
  │
  ├─► [7] DEPLOY & USE
  │     ├─► Start API server
  │     ├─► Launch dashboard
  │     └─► Make predictions
  │
  └─► END (Results in experiments/ directory)
```

---

## 📂 File Structure & Purpose

```
Bert_VS_T5/
│
├── 📁 src/                          # Source code
│   ├── data/                       # Data handling
│   │   ├── squad_loader.py         # Load SQuAD dataset
│   │   ├── xquad_loader.py         # Load XQuAD dataset
│   │   ├── mlqa_loader.py          # Load MLQA dataset
│   │   └── multilingual_preprocessor.py  # Process multilingual text
│   │
│   ├── models/                     # Model implementations
│   │   ├── mbert_wrapper.py        # mBERT model wrapper
│   │   ├── mt5_wrapper.py         # mT5 model wrapper
│   │   └── base_model.py          # Base model interface
│   │
│   ├── training/                   # Training logic
│   │   ├── zero_shot_trainer.py   # Zero-shot training
│   │   ├── few_shot_trainer.py    # Few-shot training
│   │   └── experiment_tracker.py  # Track experiments
│   │
│   ├── evaluation/                 # Evaluation tools
│   │   ├── evaluator.py           # Main evaluator
│   │   ├── metrics.py             # Calculate metrics
│   │   └── model_comparison.py    # Compare models
│   │
│   ├── inference/                  # Inference engine
│   │   ├── model_manager.py       # Manage loaded models
│   │   └── request_handler.py     # Handle prediction requests
│   │
│   └── api/                        # REST API
│       └── server.py              # FastAPI server
│
├── 📁 scripts/                     # Executable scripts
│   ├── train_zero_shot.py         # Train zero-shot model
│   ├── train_few_shot.py          # Train few-shot model
│   ├── evaluate.py                # Evaluate model
│   ├── compare_models.py          # Compare two models
│   └── download_data.py           # Download datasets
│
├── 📁 configs/                     # Configuration files
│   ├── model/                     # Model configs
│   │   ├── mbert.yaml             # mBERT settings
│   │   └── mt5.yaml               # mT5 settings
│   ├── training/                  # Training configs
│   │   ├── zero_shot.yaml         # Zero-shot settings
│   │   └── few_shot.yaml          # Few-shot settings
│   └── dataset/                    # Dataset configs
│
├── 📁 data/                        # Datasets (downloaded)
│   ├── squad/                     # SQuAD 2.0
│   ├── xquad/                     # XQuAD
│   ├── mlqa/                      # MLQA
│   └── tydiqa/                    # TyDiQA
│
├── 📁 models/                      # Trained models
│   ├── mbert/                     # mBERT checkpoints
│   └── checkpoints/               # mT5 checkpoints
│
├── 📁 experiments/                 # Results
│   ├── tracking/                  # Experiment metadata
│   └── evaluations/               # Evaluation results
│
├── 📁 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Explore data
│   ├── 02_model_training.ipynb    # Training examples
│   ├── 03_evaluation_visualization.ipynb  # Visualize results
│   └── 04_api_usage.ipynb         # API examples
│
├── 📁 logs/                        # Training logs
│
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Main documentation
```

---

## 🔀 Data Flow

### Training Flow:
```
SQuAD JSON
    │
    ▼
Data Loader (squad_loader.py)
    │
    ▼
Preprocessor (multilingual_preprocessor.py)
    │
    ▼
Model (mbert_wrapper.py or mt5_wrapper.py)
    │
    ▼
Trainer (zero_shot_trainer.py)
    │
    ▼
Checkpoint (models/checkpoints/)
    │
    ▼
Experiment Tracker (experiments/tracking/)
```

### Inference Flow:
```
User Question + Context
    │
    ▼
API Server (src/api/server.py)
    │
    ▼
Request Handler (src/inference/request_handler.py)
    │
    ▼
Model Manager (src/inference/model_manager.py)
    │
    ▼
Model (mbert_wrapper.py or mt5_wrapper.py)
    │
    ▼
Answer + Confidence
```

### Evaluation Flow:
```
Dataset (XQuAD, MLQA, etc.)
    │
    ▼
Data Loader
    │
    ▼
Evaluator (src/evaluation/evaluator.py)
    │
    ▼
Metrics Calculator (src/evaluation/metrics.py)
    │
    ▼
Results JSON (experiments/evaluations/)
```

---

## 🎓 Learning Paths

### Path 1: Quick Results (30-45 minutes)
```
Setup → Download Data → Quick Train → Evaluate → Done
```

### Path 2: Full Research (2-4 hours)
```
Setup → Download Data → Full Train → Evaluate → 
Cross-Lingual Eval → Compare Models → Analyze Results
```

### Path 3: Production Use (1-2 hours)
```
Setup → Download Data → Train → Evaluate → 
Start API → Use Dashboard → Make Predictions
```

---

## 🔑 Key Concepts Explained

### 1. **Zero-Shot Learning**
- **What**: Train model on English data only
- **Why**: Test if model can transfer knowledge to other languages
- **How**: Train on SQuAD (English), test on XQuAD (other languages)
- **Result**: Measures cross-lingual transfer capability

### 2. **Few-Shot Learning**
- **What**: Fine-tune with a few examples (1, 5, 10, or 50) per language
- **Why**: Improve performance on target languages with minimal data
- **How**: Start from zero-shot checkpoint, add few examples, fine-tune
- **Result**: Better performance than zero-shot, less data than full training

### 3. **Extractive vs Generative**
- **mBERT (Extractive)**: Finds answer span directly from context
  - Faster inference
  - Answer must exist in context
  - Example: "Paris" from "Paris is the capital of France"
  
- **mT5 (Generative)**: Generates answer text
  - More flexible
  - Can paraphrase or summarize
  - Example: Can generate "The capital city of France" even if exact phrase not in context

### 4. **Cross-Lingual QA**
- **Same Language**: Question and context in same language (e.g., English-English)
- **Cross-Lingual**: Question and context in different languages (e.g., English-Spanish)
- **Challenge**: Model must understand both languages and transfer knowledge

---

## 📊 Expected Results Structure

After running experiments, you'll have:

```
experiments/
├── tracking/
│   ├── zero_shot_mbert_20251116_181947.json
│   └── zero_shot_mt5_20251117_110556.json
│
└── evaluations/
    ├── mbert_squad_dev_20251116_182000.json
    ├── mt5_squad_dev_20251117_110600.json
    ├── mbert_xquad_es_20251116_182100.json
    └── mt5_xquad_es_20251117_110700.json
```

Each JSON file contains:
- Model configuration
- Training parameters
- Evaluation metrics (EM, F1, BLEU, ROUGE)
- Language-specific performance
- Statistical analysis

---

## 🎯 Decision Tree: What Should I Do?

```
Do you want to...
│
├─► See quick results?
│   └─► Use: train_mt5_comparison.sh (30 min)
│
├─► Train a production model?
│   └─► Use: train_zero_shot.py with full data (2-4 hours)
│
├─► Compare mBERT vs mT5?
│   └─► Train both, then use: compare_models.py
│
├─► Test cross-lingual performance?
│   └─► Train zero-shot, then evaluate on XQuAD/MLQA
│
├─► Use the system interactively?
│   └─► Start API + Dashboard
│
└─► Understand the data?
    └─► Use: notebooks/01_data_exploration.ipynb
```

---

## 💡 Pro Tips

1. **Start Small**: Use quick training script first to verify everything works
2. **Check Logs**: Training logs in `logs/` show progress and errors
3. **Save Experiments**: All results are automatically tracked
4. **Use Notebooks**: Jupyter notebooks provide interactive exploration
5. **Monitor Memory**: mT5 uses more memory than mBERT
6. **Language Pairs**: Test both same-language and cross-lingual scenarios

---

This workflow diagram should help you understand how all the pieces fit together! 🚀

