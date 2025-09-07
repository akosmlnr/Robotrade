#!/bin/bash

# Example Docker Usage Script for Stock Prediction Deep Learning
# This script demonstrates various ways to use the Docker setup

echo "🚀 Stock Prediction Deep Learning - Docker Examples"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running. Starting examples..."

# Example 1: Build the Docker image
echo ""
echo "📦 Example 1: Building Docker Image"
echo "-----------------------------------"
print_status "Building Docker image..."
make build
if [ $? -eq 0 ]; then
    print_status "✅ Docker image built successfully!"
else
    print_error "❌ Failed to build Docker image"
    exit 1
fi

# Example 2: Train with default parameters
echo ""
echo "🎯 Example 2: Training with Default Parameters"
echo "----------------------------------------------"
print_status "Training model with default parameters (GOOG stock)..."
print_warning "This may take several minutes depending on your system..."
make train
if [ $? -eq 0 ]; then
    print_status "✅ Training completed successfully!"
    print_status "Check the 'outputs/' directory for results"
else
    print_error "❌ Training failed"
fi

# Example 3: Train for different stock
echo ""
echo "🍎 Example 3: Training for Apple Stock"
echo "-------------------------------------"
print_status "Training model for AAPL stock..."
make train-aapl
if [ $? -eq 0 ]; then
    print_status "✅ AAPL training completed successfully!"
else
    print_error "❌ AAPL training failed"
fi

# Example 4: Custom training parameters
echo ""
echo "⚙️  Example 4: Custom Training Parameters"
echo "----------------------------------------"
print_status "Training with custom parameters (TSLA, 50 epochs)..."
make run-custom TICKER=TSLA EPOCHS=50
if [ $? -eq 0 ]; then
    print_status "✅ Custom training completed successfully!"
else
    print_error "❌ Custom training failed"
fi

# Example 5: Show available commands
echo ""
echo "📋 Example 5: Available Commands"
echo "-------------------------------"
print_status "Showing all available Makefile commands:"
make help

# Example 6: Check container status
echo ""
echo "📊 Example 6: Container Status"
echo "-----------------------------"
print_status "Checking Docker container status:"
make status

# Example 7: Show outputs
echo ""
echo "📁 Example 7: Generated Outputs"
echo "------------------------------"
print_status "Contents of outputs directory:"
ls -la outputs/ 2>/dev/null || print_warning "No outputs directory found"

print_status "Contents of models directory:"
ls -la models/ 2>/dev/null || print_warning "No models directory found"

print_status "Contents of data directory:"
ls -la data/ 2>/dev/null || print_warning "No data directory found"

echo ""
echo "🎉 Examples completed!"
echo "====================="
print_status "You can now:"
echo "  • Use 'make up-jupyter' to start Jupyter notebook for interactive development"
echo "  • Use 'make run-realtime' to run real-time predictions (requires trained model)"
echo "  • Use 'make clean' to clean up Docker resources"
echo "  • Check the README.md for more detailed usage instructions"
echo ""
print_status "Happy trading! 📈"
