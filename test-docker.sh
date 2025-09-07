#!/bin/bash

# Test Docker Setup for Stock Prediction
echo "🧪 Testing Docker Setup for Stock Prediction"
echo "============================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Test 1: Check if Docker is running
echo "1. Checking Docker status..."
if docker info > /dev/null 2>&1; then
    print_status "Docker is running"
else
    print_error "Docker is not running"
    exit 1
fi

# Test 2: Check if image exists
echo "2. Checking Docker image..."
if docker images | grep -q stock-prediction; then
    print_status "Docker image 'stock-prediction' exists"
else
    print_warning "Docker image not found, building..."
    make build
    if [ $? -eq 0 ]; then
        print_status "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
fi

# Test 3: Test basic container run
echo "3. Testing basic container run..."
docker run --rm stock-prediction python --version
if [ $? -eq 0 ]; then
    print_status "Container runs successfully"
else
    print_error "Container failed to run"
    exit 1
fi

# Test 4: Test Python imports
echo "4. Testing Python imports..."
docker run --rm stock-prediction python -c "
import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
print('All imports successful!')
print(f'TensorFlow version: {tf.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'NumPy version: {np.__version__}')
"
if [ $? -eq 0 ]; then
    print_status "All Python packages imported successfully"
else
    print_error "Python import test failed"
    exit 1
fi

# Test 5: Test help command
echo "5. Testing help command..."
docker run --rm stock-prediction python stock_prediction_deep_learning.py --help
if [ $? -eq 0 ]; then
    print_status "Help command works"
else
    print_error "Help command failed"
    exit 1
fi

echo ""
echo "🎉 All tests passed! Docker setup is working correctly."
echo ""
echo "Next steps:"
echo "  • Run 'make train' to train a model"
echo "  • Run 'make up-jupyter' to start Jupyter notebook"
echo "  • Run 'make help' to see all available commands"
echo ""
print_status "Docker setup is ready to use! 🚀"
