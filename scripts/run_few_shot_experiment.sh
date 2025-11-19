#!/bin/bash
# Few-shot experiment workflow: Train with different shot counts and generate learning curves

set -e  # Exit on error

echo "=========================================="
echo "Cross-Lingual QA Few-Shot Experiment"
echo "=========================================="

# Configuration
DATA_DIR="data"
XQUAD_PATH="${DATA_DIR}/xquad"
BASELINE_DIR="experiments/baseline"
OUTPUT_DIR="experiments/few_shot"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL=${1:-mbert}  # Default to mbert, can pass mt5 as argument

# Shot counts to evaluate
SHOT_COUNTS=(1 5 10 50)

# Create output directory
mkdir -p ${OUTPUT_DIR}/${MODEL}

echo ""
echo "Model: ${MODEL}"
echo "Baseline checkpoint: ${BASELINE_DIR}/${MODEL}/best_checkpoint.pt"
echo ""

# Train and evaluate for each shot count
for SHOTS in "${SHOT_COUNTS[@]}"; do
    echo "=========================================="
    echo "Training with ${SHOTS} shots"
    echo "=========================================="
    
    # Train
    python scripts/train_few_shot.py \
        --model ${MODEL} \
        --checkpoint ${BASELINE_DIR}/${MODEL}/best_checkpoint.pt \
        --data-path ${XQUAD_PATH} \
        --dataset-type xquad \
        --num-shots ${SHOTS} \
        --output-dir ${OUTPUT_DIR}/${MODEL}/${SHOTS}shot \
        --experiment-name "${MODEL}_${SHOTS}shot_${TIMESTAMP}" \
        --batch-size 8 \
        --learning-rate 1e-5 \
        --num-epochs 10 \
        --seed 42
    
    echo ""
    echo "Evaluating ${SHOTS}-shot model"
    echo "=========================================="
    
    # Evaluate
    python scripts/evaluate.py \
        --model ${MODEL} \
        --checkpoint ${OUTPUT_DIR}/${MODEL}/${SHOTS}shot/best_checkpoint.pt \
        --data-path ${XQUAD_PATH} \
        --dataset-type xquad \
        --output-dir ${OUTPUT_DIR}/${MODEL}/${SHOTS}shot/evaluations
    
    echo ""
done

echo "=========================================="
echo "Generating Learning Curves"
echo "=========================================="

# Generate learning curves (would need a separate script)
python scripts/generate_learning_curves.py \
    --model ${MODEL} \
    --results-dir ${OUTPUT_DIR}/${MODEL} \
    --output-dir ${OUTPUT_DIR}/${MODEL}/visualizations

echo ""
echo "=========================================="
echo "Few-Shot Experiment Complete!"
echo "Results saved to: ${OUTPUT_DIR}/${MODEL}"
echo "=========================================="
