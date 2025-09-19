"""
Enhanced MASTER Training Script with Polygon.io Integration
Trains MASTER model with market data, news sentiment, and trading schemes
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from feature_engineering_pipeline import MASTERFeaturePipeline
from enhanced_master import EnhancedMASTERModel
from polygon_data_fetcher import PolygonDataFetcher
from news_sentiment_processor import NewsSentimentProcessor
from trading_schemes_features import TradingSchemesFeatures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMASTERTrainer:
    """
    Enhanced MASTER trainer with Polygon.io integration
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the enhanced MASTER trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pipeline = MASTERFeaturePipeline()
        self.model = None
        self.training_history = []
        
        # Set random seeds for reproducibility
        if config.get('seed') is not None:
            np.random.seed(config['seed'])
            torch.manual_seed(config['seed'])
            torch.cuda.manual_seed_all(config['seed'])
            torch.backends.cudnn.deterministic = True
    
    def prepare_data(self, tickers: List[str], from_date: str, to_date: str, 
                    test_split: float = 0.2, val_split: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training, validation, and test data
        
        Args:
            tickers: List of stock symbols
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            test_split: Fraction of data for testing
            val_split: Fraction of data for validation
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info(f"Preparing data for {len(tickers)} tickers from {from_date} to {to_date}")
        
        # Run full feature engineering pipeline
        features, labels = self.pipeline.run_full_pipeline(
            tickers, from_date, to_date, 
            lookback_window=self.config.get('lookback_window', 8)
        )
        
        if len(features) == 0 or len(labels) == 0:
            raise ValueError("No data generated from pipeline")
        
        logger.info(f"Generated data: features {features.shape}, labels {labels.shape}")
        
        # Split data chronologically
        n_samples = len(features)
        test_size = int(n_samples * test_split)
        val_size = int(n_samples * val_split)
        train_size = n_samples - test_size - val_size
        
        # Split indices
        train_end = train_size
        val_end = train_size + val_size
        
        # Split features and labels
        X_train = features[:train_end]
        y_train = labels[:train_end]
        X_val = features[train_end:val_end]
        y_val = labels[train_end:val_end]
        X_test = features[val_end:]
        y_test = labels[val_end:]
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                          batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self):
        """Initialize the enhanced MASTER model"""
        logger.info("Initializing enhanced MASTER model...")
        
        # Calculate feature dimensions
        d_feat = self.config.get('d_feat', 200)
        gate_input_start_index = self.config.get('gate_input_start_index', 200)
        gate_input_end_index = self.config.get('gate_input_end_index', 263)
        
        # Initialize model
        self.model = EnhancedMASTERModel(
            d_feat=d_feat,
            d_model=self.config.get('d_model', 256),
            t_nhead=self.config.get('t_nhead', 4),
            s_nhead=self.config.get('s_nhead', 2),
            gate_input_start_index=gate_input_start_index,
            gate_input_end_index=gate_input_end_index,
            T_dropout_rate=self.config.get('dropout', 0.5),
            S_dropout_rate=self.config.get('dropout', 0.5),
            beta=self.config.get('beta', 5),
            num_tasks=self.config.get('num_tasks', 8),
            task_weights=self.config.get('task_weights', [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
            n_epochs=self.config.get('n_epochs', 10),
            lr=self.config.get('lr', 1e-5),
            GPU=self.config.get('GPU', 0),
            train_stop_loss_thred=self.config.get('train_stop_loss_thred', 0.95),
            seed=self.config.get('seed', 42)
        )
        
        logger.info("Model initialized successfully!")
    
    def train_model(self, train_loader, val_loader):
        """
        Train the enhanced MASTER model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("Starting model training...")
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        # Training loop
        best_val_loss = float('inf')
        patience = self.config.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(self.config.get('n_epochs', 10)):
            # Training
            train_loss = self._train_epoch(train_loader)
            
            # Validation
            val_loss = self._validate_epoch(val_loader)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.config.get('n_epochs', 10)} - "
                       f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save training history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_model(f"best_model_epoch_{epoch+1}.pkl")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training completed!")
    
    def _train_epoch(self, train_loader):
        """Train for one epoch"""
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
        
        return np.mean(losses)
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.model.eval()
        losses = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.model.device)
                batch_y = batch_y.to(self.model.device)
                
                pred = self.model.model(batch_X)
                loss = self.model.multi_task_loss_fn(pred, batch_y)
                losses.append(loss.item())
        
        return np.mean(losses)
    
    def evaluate_model(self, test_loader):
        """
        Evaluate the model on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        self.model.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.model.device)
                batch_y = batch_y.to(self.model.device)
                
                pred = self.model.model(batch_X)
                
                # Extract primary prediction (return)
                primary_pred = pred[:, 0].cpu().numpy()
                primary_label = batch_y[:, 0].cpu().numpy()
                
                all_predictions.append(primary_pred)
                all_labels.append(primary_label)
        
        # Combine predictions and labels
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, labels)
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def _calculate_metrics(self, predictions, labels):
        """Calculate evaluation metrics"""
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(labels))
        pred_clean = predictions[mask]
        label_clean = labels[mask]
        
        if len(pred_clean) == 0:
            return {'IC': 0, 'ICIR': 0, 'RIC': 0, 'RICIR': 0, 'MSE': 0, 'MAE': 0}
        
        # Information Coefficient
        ic = np.corrcoef(pred_clean, label_clean)[0, 1] if len(pred_clean) > 1 else 0
        
        # Rank Information Coefficient
        from scipy.stats import spearmanr
        ric, _ = spearmanr(pred_clean, label_clean) if len(pred_clean) > 1 else (0, 0)
        
        # Mean Squared Error
        mse = np.mean((pred_clean - label_clean) ** 2)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(pred_clean - label_clean))
        
        # ICIR and RICIR
        icir = ic / np.std(pred_clean) if np.std(pred_clean) > 0 else 0
        ricir = ric / np.std(pred_clean) if np.std(pred_clean) > 0 else 0
        
        return {
            'IC': ic,
            'ICIR': icir,
            'RIC': ric,
            'RICIR': ricir,
            'MSE': mse,
            'MAE': mae
        }
    
    def _save_model(self, filename: str):
        """Save model to file"""
        if self.model is not None:
            torch.save(self.model.model.state_dict(), filename)
            logger.info(f"Model saved to {filename}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        df_history = pd.DataFrame(self.training_history)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(df_history['epoch'], df_history['train_loss'], label='Train Loss')
        plt.plot(df_history['epoch'], df_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(df_history['epoch'], df_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Over Time')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def run_full_training(self, tickers: List[str], from_date: str, to_date: str):
        """
        Run the complete training pipeline
        
        Args:
            tickers: List of stock symbols
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        """
        logger.info("Starting full training pipeline...")
        
        try:
            # Step 1: Prepare data
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(
                tickers, from_date, to_date
            )
            
            # Step 2: Create data loaders
            train_loader, val_loader, test_loader = self.create_data_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size=self.config.get('batch_size', 32)
            )
            
            # Step 3: Initialize model
            self.initialize_model()
            
            # Step 4: Train model
            self.train_model(train_loader, val_loader)
            
            # Step 5: Evaluate model
            test_metrics = self.evaluate_model(test_loader)
            
            # Step 6: Plot results
            self.plot_training_history('training_history.png')
            
            logger.info("Full training pipeline completed successfully!")
            logger.info(f"Final test metrics: {test_metrics}")
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Configuration
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
        'GPU': 0,
        'train_stop_loss_thred': 0.95,
        'patience': 5,
        'batch_size': 32,
        'lookback_window': 8,
        'seed': 42
    }
    
    # Example tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
    from_date = '2024-01-01'
    to_date = '2024-03-31'
    
    # Initialize trainer
    trainer = EnhancedMASTERTrainer(config)
    
    # Run full training
    print("Starting enhanced MASTER training...")
    metrics = trainer.run_full_training(tickers, from_date, to_date)
    
    print(f"Training completed! Final metrics: {metrics}")
