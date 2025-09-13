# Real-time LSTM Prediction System Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/logs

# Set permissions
RUN chmod +x core/cli.py

# Expose ports for API server
EXPOSE 5000

# Default command - show help
CMD ["python", "-m", "core.cli", "--help"]
