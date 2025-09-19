"""
Enhanced MASTER Training Script for Windows RTX GPUs
Optimized for Windows environment with RTX 30/40 series support
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import gc
from typing import Dict, List, Tuple, Optional

# Windows-specific optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable for performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

# Add MASTER to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our modules
from train_enhanced_master import EnhancedMASTERTrainer
from feature_engineering_pipeline import MASTERFeaturePipeline
from enhanced_master import EnhancedMASTERModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WindowsRTXTrainer(EnhancedMASTERTrainer):
    """
    Windows RTX-optimized MASTER trainer
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.setup_windows_optimizations()
    
    def setup_windows_optimizations(self):
        """Setup Windows-specific optimizations for RTX GPUs"""
        if torch.cuda.is_available():
            # RTX 30/40 series optimizations
            device_props = torch.cuda.get_device_properties(0)
            if device_props.major >= 8:  # RTX 30/40 series
                # Enable Tensor Cores
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("RTX 30/40 series optimizations enabled")
            
            # Windows-specific settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # For performance
            
            # Memory management
            torch.cuda.set_per_process_memory_fraction(0.9)
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM on Windows"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
    
    def train_epoch(self, train_loader):
        """Windows-optimized training epoch"""
        self.model.model.train()
        losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.model.device)
            batch_y = batch_y.to(self.model.device)
            
            # Forward pass
            pred = self.model.model(batch_X)
            loss = self.model.multi_task_loss_fn(pred, batch_y)
            
            # Backward pass
            self.model.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.model.parameters(), 3.0)
            self.model.train_optimizer.step()
            
            losses.append(loss.item())
            
            # Clear memory periodically
            if len(losses) % 10 == 0:
                self.clear_gpu_memory()
        
        return np.mean(losses)
    
    def _validate_epoch(self, val_loader):
        """Windows-optimized validation epoch"""
        self.model.model.eval()
        losses = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.model.device)
                batch_y = batch_y.to(self.model.device)
                
                pred = self.model.model(batch_X)
                loss = self.model.multi_task_loss_fn(pred, batch_y)
                losses.append(loss.item())
                
                # Clear memory periodically
                if len(losses) % 10 == 0:
                    self.clear_gpu_memory()
        
        return np.mean(losses)
    
    def monitor_gpu_usage(self):
        """Monitor GPU usage for Windows RTX"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def run_full_training(self, tickers: List[str], from_date: str, to_date: str):
        """Run full training with Windows RTX optimizations"""
        logger.info("Starting Enhanced MASTER training on Windows RTX...")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.monitor_gpu_usage()
        
        try:
            # Step 1: Prepare data
            logger.info("Preparing data...")
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(
                tickers, from_date, to_date
            )
            
            # Step 2: Create data loaders with Windows optimizations
            logger.info("Creating data loaders...")
            train_loader, val_loader, test_loader = self.create_data_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size=self.config.get('batch_size', 16)  # RTX-optimized
            )
            
            # Step 3: Initialize model
            logger.info("Initializing model...")
            self.initialize_model()
            
            # Step 4: Train model
            logger.info("Starting training...")
            self.train_model(train_loader, val_loader)
            
            # Step 5: Evaluate model
            logger.info("Evaluating model...")
            test_metrics = self.evaluate_model(test_loader)
            
            # Step 6: Plot results
            self.plot_training_history('training_history_windows.png')
            
            logger.info("Training completed successfully!")
            logger.info(f"Final test metrics: {test_metrics}")
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Final memory cleanup
            self.clear_gpu_memory()


def get_rtx_optimized_config():
    """Get RTX-optimized configuration for Windows"""
    return {
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


def main():
    """Main training function for Windows RTX"""
    print("=" * 80)
    print("ENHANCED MASTER MODEL - WINDOWS RTX TRAINING")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Training will use CPU (slower).")
        print("Please install CUDA toolkit and PyTorch with CUDA support.")
    else:
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Get RTX-optimized configuration
    config = get_rtx_optimized_config()
    
    # Example tickers (adjust as needed)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    from_date = '2024-01-01'
    to_date = '2024-03-31'
    
    print(f"\nTraining configuration:")
    print(f"- Tickers: {tickers}")
    print(f"- Date range: {from_date} to {to_date}")
    print(f"- Batch size: {config['batch_size']} (RTX-optimized)")
    print(f"- Model dimension: {config['d_model']}")
    print(f"- Epochs: {config['n_epochs']}")
    
    # Initialize Windows RTX trainer
    trainer = WindowsRTXTrainer(config)
    
    # Run training
    print("\n🚀 Starting training...")
    metrics = trainer.run_full_training(tickers, from_date, to_date)
    
    if metrics:
        print("\n✅ Training completed successfully!")
        print(f"Final metrics: {metrics}")
    else:
        print("\n❌ Training failed. Check logs for details.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
