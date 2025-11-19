#!/bin/bash
# Script to test mBERT training results

cd /Users/aditya/Downloads/Bert_VS_T5
source venv/bin/activate

echo "=========================================="
echo "Testing mBERT Training Results"
echo "=========================================="
echo ""

# Check if checkpoint exists
CHECKPOINT="models/mbert_retrained/best_model.pt"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="models/mbert/best_model.pt"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: No mBERT checkpoint found!"
    echo "   Looked for:"
    echo "   - models/mbert_retrained/best_model.pt"
    echo "   - models/mbert/best_model.pt"
    exit 1
fi

echo "✅ Found checkpoint: $CHECKPOINT"
echo ""

# Test 1: Evaluate on SQuAD dev set (English, same language)
echo "=========================================="
echo "Test 1: SQuAD Dev Set (English-English)"
echo "=========================================="
echo "This tests performance on English (same language as training)"
echo ""

python scripts/evaluate.py \
    --model mbert \
    --checkpoint "$CHECKPOINT" \
    --data-path data/squad/dev-v2.0.json \
    --dataset-type squad \
    --output-dir experiments/evaluations

echo ""
echo "✅ Test 1 Complete!"
echo ""

# Test 2: Quick test on small subset (optional, faster)
read -p "Do you want to test on a smaller subset first? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Creating small test subset..."
    # This would require a script to create subset, skip for now
    echo "Skipping subset test - running full evaluation"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Check results in: experiments/evaluations/"
echo "2. For cross-lingual testing, run:"
echo "   python scripts/evaluate.py --model mbert --checkpoint $CHECKPOINT --data-path data/xquad/xquad-master/xquad.es.json --dataset-type xquad"
echo ""
echo "3. Compare with previous results:"
echo "   python scripts/compare_models.py --results-a experiments/evaluations/mbert_squad_*.json --results-b experiments/evaluations/mbert_squad_20251117_230415.json --model-a-name 'mBERT Retrained' --model-b-name 'mBERT Original'"
echo ""

