# Dockerfile for Cross-Lingual QA System

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data models experiments logs cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-m", "src.api.server"]
