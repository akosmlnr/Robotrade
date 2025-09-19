# Enhanced MASTER Model - Windows RTX Setup Guide

## 🚀 **Windows RTX GPU Setup for Enhanced MASTER Model**

### **Prerequisites**

1. **Python 3.8+** (Recommended: Python 3.9 or 3.10)
2. **CUDA Toolkit 11.8+** (for RTX 30/40 series)
3. **cuDNN 8.6+** (for optimal performance)
4. **Polygon.io API Key**

---

## **Step 1: Environment Setup**

### **1.1 Create Virtual Environment**
```cmd
# Navigate to Robotrade directory
cd C:\path\to\Robotrade

# Create virtual environment
python -m venv venv_master

# Activate virtual environment
venv_master\Scripts\activate
```

### **1.2 Install PyTorch with CUDA Support**
```cmd
# For RTX 30/40 series (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### **1.3 Install Enhanced MASTER Dependencies**
```cmd
# Install core dependencies
pip install -r model/MASTER/requirements_enhanced.txt

# Install additional Windows-specific packages
pip install python-dotenv requests pandas numpy scipy scikit-learn
pip install transformers torch-audio
pip install matplotlib seaborn plotly
pip install jupyter ipykernel
```

---

## **Step 2: CUDA and GPU Configuration**

### **2.1 Verify RTX GPU Detection**
```python
# Test script: test_gpu.py
import torch
import torch.nn as nn

print("=== GPU Configuration ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
```

### **2.2 RTX-Specific Optimizations**
```python
# Add to your training script
import torch

# Enable cuDNN benchmarking for consistent input sizes
torch.backends.cudnn.benchmark = True

# Enable cuDNN deterministic mode for reproducibility
torch.backends.cudnn.deterministic = True

# Set memory fraction to avoid OOM
torch.cuda.set_per_process_memory_fraction(0.9)
```

---

## **Step 3: Environment Configuration**

### **3.1 Create .env File**
```cmd
# Create .env file in Robotrade root directory
echo POLYGON_API_KEY=your_polygon_api_key_here > .env
echo CUDA_VISIBLE_DEVICES=0 >> .env
echo PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 >> .env
```

### **3.2 Windows-Specific Configuration**
```python
# Add to your training script
import os
import torch

# Windows-specific optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable for performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

# Set Windows-specific paths
if os.name == 'nt':  # Windows
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # For performance
```

---

## **Step 4: Training Configuration**

### **4.1 RTX-Optimized Configuration**
```python
# RTX-optimized config for Windows
config = {
    # Model Architecture
    'd_feat': 200,
    'd_model': 256,  # Reduced for RTX memory
    't_nhead': 4,
    's_nhead': 2,
    'dropout': 0.5,
    
    # Feature Gates
    'gate_input_start_index': 200,
    'gate_input_end_index': 263,
    'beta': 5,
    
    # Multi-Task Learning
    'num_tasks': 8,
    'task_weights': [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    
    # RTX-Optimized Training
    'n_epochs': 20,
    'lr': 1e-5,
    'batch_size': 16,  # Reduced for RTX memory
    'patience': 5,
    'lookback_window': 8,
    
    # GPU Configuration
    'GPU': 0,  # Use first GPU
    'seed': 42,
    
    # Windows-specific
    'num_workers': 0,  # Windows multiprocessing
    'pin_memory': True,
    'persistent_workers': False
}
```

### **4.2 Memory Management for RTX**
```python
# Add to training script
import torch
import gc

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Use in training loop
for epoch in range(n_epochs):
    # Training code...
    clear_gpu_memory()  # Clear memory after each epoch
```

---

## **Step 5: Running the Training**

### **5.1 Basic Training Script**
```python
# train_master_windows.py
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add MASTER to path
sys.path.append(str(Path(__file__).parent / "model" / "MASTER"))

from train_enhanced_master import EnhancedMASTERTrainer

def main():
    # Windows-specific setup
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # RTX-optimized configuration
    config = {
        'd_feat': 200,
        'd_model': 256,
        't_nhead': 4,
        's_nhead': 2,
        'dropout': 0.5,
        'gate_input_start_index': 200,
        'gate_input_end_index': 263,
        'beta': 5,
        'num_tasks': 8,
        'task_weights': [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        'n_epochs': 20,
        'lr': 1e-5,
        'batch_size': 16,  # RTX-optimized
        'patience': 5,
        'lookback_window': 8,
        'GPU': 0,
        'seed': 42,
        'num_workers': 0,  # Windows
        'pin_memory': True
    }
    
    # Example tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    from_date = '2024-01-01'
    to_date = '2024-03-31'
    
    print("Starting Enhanced MASTER training on Windows RTX...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Initialize trainer
    trainer = EnhancedMASTERTrainer(config)
    
    # Run training
    try:
        metrics = trainer.run_full_training(tickers, from_date, to_date)
        print(f"Training completed! Metrics: {metrics}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

### **5.2 Command Line Execution**
```cmd
# Activate virtual environment
venv_master\Scripts\activate

# Run training
python model/MASTER/train_enhanced_master.py

# Or run custom script
python train_master_windows.py
```

---

## **Step 6: RTX-Specific Optimizations**

### **6.1 Memory Optimization**
```python
# Add to training script
import torch

# RTX memory optimizations
torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for performance

# Mixed precision training for RTX
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# Use in training loop
with autocast():
    pred = model(features)
    loss = criterion(pred, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### **6.2 RTX 30/40 Series Specific**
```python
# RTX 30/40 series optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major >= 8:  # RTX 30/40 series
        # Enable Tensor Cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Use mixed precision
        use_amp = True
    else:
        use_amp = False
```

---

## **Step 7: Troubleshooting**

### **7.1 Common Windows Issues**

**CUDA Out of Memory:**
```python
# Reduce batch size
config['batch_size'] = 8  # or 4

# Clear memory
torch.cuda.empty_cache()
```

**DLL Loading Issues:**
```cmd
# Install Visual C++ Redistributable
# Download from Microsoft website
```

**Permission Issues:**
```cmd
# Run as Administrator if needed
# Check antivirus exclusions
```

### **7.2 RTX-Specific Issues**

**Driver Compatibility:**
```cmd
# Update NVIDIA drivers to latest version
# Use CUDA 11.8+ for RTX 30/40 series
```

**Memory Fragmentation:**
```python
# Add to training script
import gc
import torch

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

---

## **Step 8: Performance Monitoring**

### **8.1 GPU Monitoring**
```python
# Add to training script
import torch

def monitor_gpu():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB")
        print(f"GPU Utilization: {torch.cuda.utilization()}%")
```

### **8.2 Training Progress**
```python
# Add progress tracking
from tqdm import tqdm

for epoch in tqdm(range(n_epochs), desc="Training"):
    # Training code...
    monitor_gpu()  # Monitor GPU usage
```

---

## **Quick Start Commands**

```cmd
# 1. Setup environment
python -m venv venv_master
venv_master\Scripts\activate

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r model/MASTER/requirements_enhanced.txt

# 4. Create .env file
echo POLYGON_API_KEY=your_key_here > .env

# 5. Run training
python model/MASTER/train_enhanced_master.py
```

This guide provides everything you need to run the enhanced MASTER model on Windows with RTX GPU support! 🚀
