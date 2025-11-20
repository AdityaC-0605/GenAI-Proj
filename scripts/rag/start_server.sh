#!/bin/bash
# Quick start script for RAG API server

set -e

echo "========================================="
echo "RAG API Server Startup"
echo "========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating your venv first"
    echo ""
fi

# Configuration
CONFIG_NAME=${CONFIG_NAME:-"default"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
WORKERS=${WORKERS:-"1"}
RATE_LIMIT=${RATE_LIMIT:-"100"}

echo "Configuration:"
echo "  Config: $CONFIG_NAME"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Rate Limit: $RATE_LIMIT req/min"
echo ""

# Check if vector index exists
if [ ! -d "data/vector_index" ]; then
    echo "❌ Error: Vector index not found at data/vector_index/"
    echo ""
    echo "Please create an index first:"
    echo "  python scripts/rag/create_vector_index.py \\"
    echo "    --dataset data/squad/train-v2.0.json \\"
    echo "    --dataset-type squad"
    echo ""
    exit 1
fi

echo "✓ Vector index found"
echo ""

# Start server
echo "Starting RAG API server..."
echo "API docs will be available at: http://$HOST:$PORT/docs"
echo ""

export CONFIG_NAME
export RATE_LIMIT

uvicorn src.api.rag_server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info
