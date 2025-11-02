# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY src/ ./src/

# Create directory for MLflow artifacts (if needed)
RUN mkdir -p ./mlruns

# Set environment variables
ENV PORT=8080
ENV MODEL_NAME=iris-classifier
ENV MODEL_STAGE=Production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}