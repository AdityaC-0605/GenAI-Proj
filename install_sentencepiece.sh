#!/bin/bash
# Install SentencePiece for mT5 training

echo "📦 Installing SentencePiece library..."
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "   Activating venv..."
    source venv/bin/activate
fi

# Install sentencepiece
pip install sentencepiece

echo ""
echo "✅ SentencePiece installed successfully!"
echo ""
echo "Now you can train mT5:"
echo "  ./train_with_caffeinate.sh mt5 data/squad/train-v2.0.json"
