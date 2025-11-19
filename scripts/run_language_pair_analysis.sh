#!/bin/bash
# Language pair analysis workflow: Analyze performance by language pair categories

set -e  # Exit on error

echo "=========================================="
echo "Language Pair Analysis Workflow"
echo "=========================================="

# Configuration
DATA_DIR="data"
XQUAD_PATH="${DATA_DIR}/xquad"
MLQA_PATH="${DATA_DIR}/mlqa"
MODEL_DIR="experiments/baseline"
OUTPUT_DIR="experiments/language_pair_analysis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Define language pair categories
# High-resource to high-resource
HIGH_HIGH_PAIRS="en-en es-es fr-fr de-de"

# High-resource to low-resource
HIGH_LOW_PAIRS="en-hi en-ja en-ko"

# Similar language families (Romance)
SIMILAR_PAIRS="es-fr fr-es"

# Distant language families
DISTANT_PAIRS="en-zh en-ar en-ja"

echo ""
echo "Step 1: Evaluate mBERT on High-to-High Resource Pairs"
echo "=========================================="
python scripts/evaluate.py \
    --model mbert \
    --checkpoint ${MODEL_DIR}/mbert/best_checkpoint.pt \
    --data-path ${XQUAD_PATH} \
    --dataset-type xquad \
    --language-pairs ${HIGH_HIGH_PAIRS} \
    --output-dir ${OUTPUT_DIR}/high_high

echo ""
echo "Step 2: Evaluate mBERT on High-to-Low Resource Pairs"
echo "=========================================="
python scripts/evaluate.py \
    --model mbert \
    --checkpoint ${MODEL_DIR}/mbert/best_checkpoint.pt \
    --data-path ${XQUAD_PATH} \
    --dataset-type xquad \
    --language-pairs ${HIGH_LOW_PAIRS} \
    --output-dir ${OUTPUT_DIR}/high_low

echo ""
echo "Step 3: Evaluate mBERT on Similar Language Families"
echo "=========================================="
python scripts/evaluate.py \
    --model mbert \
    --checkpoint ${MODEL_DIR}/mbert/best_checkpoint.pt \
    --data-path ${XQUAD_PATH} \
    --dataset-type xquad \
    --language-pairs ${SIMILAR_PAIRS} \
    --output-dir ${OUTPUT_DIR}/similar

echo ""
echo "Step 4: Evaluate mBERT on Distant Language Families"
echo "=========================================="
python scripts/evaluate.py \
    --model mbert \
    --checkpoint ${MODEL_DIR}/mbert/best_checkpoint.pt \
    --data-path ${XQUAD_PATH} \
    --dataset-type xquad \
    --language-pairs ${DISTANT_PAIRS} \
    --output-dir ${OUTPUT_DIR}/distant

echo ""
echo "Step 5: Generate Category Analysis Report"
echo "=========================================="
python scripts/analyze_language_categories.py \
    --results-dir ${OUTPUT_DIR} \
    --output-dir ${OUTPUT_DIR}/reports

echo ""
echo "=========================================="
echo "Language Pair Analysis Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
