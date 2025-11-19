#!/bin/bash
# Baseline experiment workflow: Zero-shot training and evaluation for both models

set -e  # Exit on error

echo "=========================================="
echo "Cross-Lingual QA Baseline Experiment"
echo "=========================================="

# Configuration
DATA_DIR="data"
SQUAD_PATH="${DATA_DIR}/squad/train-v2.0.json"
XQUAD_PATH="${DATA_DIR}/xquad"
OUTPUT_DIR="experiments/baseline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo ""
echo "Step 1: Training mBERT (Zero-Shot)"
echo "=========================================="
python scripts/train_zero_shot.py \
    --model mbert \
    --data-path ${SQUAD_PATH} \
    --output-dir ${OUTPUT_DIR}/mbert \
    --experiment-name "baseline_mbert_${TIMESTAMP}" \
    --batch-size 16 \
    --learning-rate 3e-5 \
    --num-epochs 3 \
    --seed 42

echo ""
echo "Step 2: Training mT5 (Zero-Shot)"
echo "=========================================="
python scripts/train_zero_shot.py \
    --model mt5 \
    --data-path ${SQUAD_PATH} \
    --output-dir ${OUTPUT_DIR}/mt5 \
    --experiment-name "baseline_mt5_${TIMESTAMP}" \
    --batch-size 8 \
    --learning-rate 3e-5 \
    --num-epochs 3 \
    --seed 42

echo ""
echo "Step 3: Evaluating mBERT on XQuAD"
echo "=========================================="
python scripts/evaluate.py \
    --model mbert \
    --checkpoint ${OUTPUT_DIR}/mbert/best_checkpoint.pt \
    --data-path ${XQUAD_PATH} \
    --dataset-type xquad \
    --output-dir ${OUTPUT_DIR}/evaluations

echo ""
echo "Step 4: Evaluating mT5 on XQuAD"
echo "=========================================="
python scripts/evaluate.py \
    --model mt5 \
    --checkpoint ${OUTPUT_DIR}/mt5/best_checkpoint.pt \
    --data-path ${XQUAD_PATH} \
    --dataset-type xquad \
    --output-dir ${OUTPUT_DIR}/evaluations \
    --include-generative-metrics

echo ""
echo "Step 5: Comparing Models"
echo "=========================================="
python scripts/compare_models.py \
    --results-a ${OUTPUT_DIR}/evaluations/mbert_xquad_*.json \
    --results-b ${OUTPUT_DIR}/evaluations/mt5_xquad_*.json \
    --model-a-name "mBERT (Zero-Shot)" \
    --model-b-name "mT5 (Zero-Shot)" \
    --output-dir ${OUTPUT_DIR}/comparisons \
    --generate-visualizations

echo ""
echo "=========================================="
echo "Baseline Experiment Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
