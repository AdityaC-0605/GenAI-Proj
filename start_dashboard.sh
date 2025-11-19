#!/bin/bash
# Quick start script for Streamlit Dashboard

echo "🚀 Starting Cross-Lingual QA Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Streamlit not found. Installing..."
    pip install streamlit plotly pandas numpy
    echo ""
fi

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Run: source venv/bin/activate"
    echo ""
fi

# Suppress urllib3 warnings (harmless on macOS)
export PYTHONWARNINGS="ignore::UserWarning:urllib3"

# Start the dashboard
echo "🌐 Opening dashboard at http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run app.py
