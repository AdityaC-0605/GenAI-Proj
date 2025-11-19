#!/bin/bash
# Start the FastAPI server for Cross-Lingual QA

echo "🚀 Starting Cross-Lingual QA API Server..."
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    echo ""
fi

# Suppress urllib3 warnings (harmless on macOS)
export PYTHONWARNINGS="ignore::UserWarning:urllib3"

# Start the API server
echo "🌐 Starting API server at http://localhost:8000"
echo "   API docs available at http://localhost:8000/docs"
echo "   Press Ctrl+C to stop"
echo ""

python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
