"""
Advanced Model Trainer for LSTM Stock Prediction System
Trains models using all available Polygon API fields for enhanced predictions
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_fetcher import PolygonDataFetcher
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Advanced model trainer that uses all available Polygon API fields
    for comprehensive stock prediction models
    """
    
    def __init__(self, models_dir: str = "lstms", api_key: str = None):
        """
        Initialize the model trainer
        
        Args:
            models_dir: Directory to save trained models
            api_key: Polygon.io API key (if None, will use environment variable)
        """
        self.models_dir = models_dir
        self.data_fetcher = PolygonDataFetcher(api_key=api_key)
        self.feature_engineer = FeatureEngineer()
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info(f"ModelTrainer initialized with models directory: {models_dir}")
    
    def prepare_training_data(self, symbol: str, lookback_days: int = 365) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare comprehensive training data using all available fields
        
        Args:
            symbol: Stock symbol to train on
            lookback_days: Number of days of historical data to use
            
        Returns:
            Tuple of (X, y) training data
        """
        try:
            logger.info(f"Preparing training data for {symbol} using {lookback_days} days")
            
            # Fetch comprehensive historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            
            # Get data with all fields (OHLCV + transactions + VWAP)
            raw_data = self.data_fetcher.fetch_15min_data(symbol, start_time, end_time)
            
            if raw_data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            logger.info(f"Retrieved {len(raw_data)} data points for {symbol}")
            
            # Apply comprehensive feature engineering
            feature_data = self.feature_engineer.add_features(raw_data)
            
            if feature_data.empty:
                raise ValueError(f"Feature engineering failed for {symbol}")
            
            logger.info(f"Created {len(feature_data.columns)} features: {list(feature_data.columns)}")
            
            # Prepare sequences for LSTM
            sequence_length = 60  # 60 time steps (15 hours of 15-min data)
            prediction_length = 1  # Predict next price
            
            X, y = self._create_sequences(feature_data, sequence_length, prediction_length)
            
            logger.info(f"Created {len(X)} training sequences with shape {X.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data for {symbol}: {e}")
            raise
    
    def _create_sequences(self, data: pd.DataFrame, sequence_length: int, prediction_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Feature DataFrame
            sequence_length: Length of input sequences
            prediction_length: Length of prediction target
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # Get feature names
            feature_names = self.feature_engineer.get_feature_names()
            available_features = [f for f in feature_names if f in data.columns]
            
            if not available_features:
                raise ValueError("No features available for sequence creation")
            
            # Extract feature data
            feature_data = data[available_features].values
            
            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_data) - prediction_length + 1):
                # Input sequence
                X.append(scaled_data[i-sequence_length:i])
                # Target (close price prediction)
                target_idx = available_features.index('close') if 'close' in available_features else 0
                y.append(scaled_data[i+prediction_length-1, target_idx])
            
            X = np.array(X)
            y = np.array(y)
            
            # Store scaler for later use
            self.scaler = scaler
            self.feature_names = available_features
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            raise
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build comprehensive LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras model
        """
        try:
            logger.info(f"Building LSTM model with input shape: {input_shape}")
            
            # Build model with multiple LSTM layers for complex patterns
            model = tf.keras.Sequential([
                # First LSTM layer with dropout
                tf.keras.layers.LSTM(
                    units=128, 
                    return_sequences=True, 
                    input_shape=input_shape,
                    dropout=0.2,
                    recurrent_dropout=0.2
                ),
                
                # Second LSTM layer
                tf.keras.layers.LSTM(
                    units=64, 
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.2
                ),
                
                # Third LSTM layer
                tf.keras.layers.LSTM(
                    units=32, 
                    return_sequences=False,
                    dropout=0.2
                ),
                
                # Dense layers for final prediction
                tf.keras.layers.Dense(units=16, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(units=8, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=1, activation='linear')  # Linear for price prediction
            ])
            
            # Compile model with appropriate optimizer and loss
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            logger.info("LSTM model built and compiled successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise
    
    def train_model(self, symbol: str, epochs: int = 100, batch_size: int = 32, 
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train comprehensive LSTM model for a symbol
        
        Args:
            symbol: Stock symbol to train
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            logger.info(f"Starting comprehensive model training for {symbol}")
            
            # Prepare training data
            X, y = self.prepare_training_data(symbol)
            
            if len(X) == 0:
                raise ValueError(f"No training data available for {symbol}")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, shuffle=False
            )
            
            logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
            
            # Build model
            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks for better training
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            logger.info("Starting model training...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            train_pred = model.predict(X_train, verbose=0)
            val_pred = model.predict(X_val, verbose=0)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, train_pred.flatten())
            val_metrics = self._calculate_metrics(y_val, val_pred.flatten())
            
            # Save model and components
            model_dir = os.path.join(self.models_dir, symbol)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model weights
            model_path = os.path.join(model_dir, "model_weights.keras")
            model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Create comprehensive model configuration
            config = {
                "symbol": symbol,
                "model_type": "LSTM_Comprehensive",
                "sequence_length": X_train.shape[1],
                "prediction_length": 1,
                "features": self.feature_names,
                "num_features": len(self.feature_names),
                "model_architecture": {
                    "input_shape": model.input_shape,
                    "total_params": model.count_params(),
                    "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
                    "non_trainable_params": sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
                },
                "scaling": {
                    "scaler_type": type(self.scaler).__name__,
                    "feature_range": self.scaler.feature_range
                },
                "training_info": {
                    "created_at": datetime.now().isoformat(),
                    "data_source": "Polygon.io API",
                    "timeframe": "15min",
                    "lookback_days": 365,
                    "training_samples": len(X_train),
                    "validation_samples": len(X_val),
                    "epochs_trained": len(history.history['loss']),
                    "batch_size": batch_size
                },
                "performance_metrics": {
                    "train_rmse": train_metrics['rmse'],
                    "train_mae": train_metrics['mae'],
                    "train_r2": train_metrics['r2'],
                    "val_rmse": val_metrics['rmse'],
                    "val_mae": val_metrics['mae'],
                    "val_r2": val_metrics['r2'],
                    "final_val_loss": history.history['val_loss'][-1],
                    "best_val_loss": min(history.history['val_loss'])
                },
                "prediction_settings": {
                    "confidence_threshold": 0.6,
                    "validation_threshold": 0.02,
                    "min_profit_percent": 2.0
                },
                "features_used": {
                    "basic_ohlcv": ['open', 'high', 'low', 'close', 'volume'],
                    "polygon_extended": ['transactions', 'vwap'],
                    "technical_indicators": ['rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower'],
                    "derived_features": ['price_change', 'volume_change', 'price_volume_ratio']
                }
            }
            
            # Save configuration
            config_path = os.path.join(model_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save training history
            history_path = os.path.join(model_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history.history, f, indent=2)
            
            logger.info(f"Model training completed for {symbol}")
            logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")
            logger.info(f"Validation R²: {val_metrics['r2']:.4f}")
            
            return {
                'success': True,
                'symbol': symbol,
                'model_path': model_path,
                'config_path': config_path,
                'training_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'config': config
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        try:
            # Remove any NaN or infinite values
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -1.0}
            
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            r2 = r2_score(y_true_clean, y_pred_clean)
            
            return {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -1.0}
    
    def train_multiple_symbols(self, symbols: List[str], **training_kwargs) -> Dict[str, Any]:
        """
        Train models for multiple symbols
        
        Args:
            symbols: List of stock symbols to train
            **training_kwargs: Additional training parameters
            
        Returns:
            Dictionary with results for each symbol
        """
        try:
            logger.info(f"Training models for {len(symbols)} symbols: {symbols}")
            
            results = {}
            successful_trains = 0
            
            for symbol in symbols:
                logger.info(f"Training model for {symbol}...")
                
                try:
                    result = self.train_model(symbol, **training_kwargs)
                    results[symbol] = result
                    
                    if result['success']:
                        successful_trains += 1
                        logger.info(f"✅ Successfully trained model for {symbol}")
                    else:
                        logger.error(f"❌ Failed to train model for {symbol}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"❌ Exception during training for {symbol}: {e}")
                    results[symbol] = {
                        'success': False,
                        'symbol': symbol,
                        'error': str(e)
                    }
            
            summary = {
                'total_symbols': len(symbols),
                'successful_trains': successful_trains,
                'failed_trains': len(symbols) - successful_trains,
                'results': results
            }
            
            logger.info(f"Training summary: {successful_trains}/{len(symbols)} models trained successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch training: {e}")
            return {
                'total_symbols': len(symbols),
                'successful_trains': 0,
                'failed_trains': len(symbols),
                'error': str(e)
            }


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the model trainer
    trainer = ModelTrainer()
    
    # Train a single model
    symbol = "AAPL"
    print(f"Training comprehensive model for {symbol}...")
    
    try:
        result = trainer.train_model(symbol, epochs=50, batch_size=64)
        
        if result['success']:
            print(f"✅ Model training successful for {symbol}")
            print(f"Validation RMSE: {result['validation_metrics']['rmse']:.4f}")
            print(f"Validation R²: {result['validation_metrics']['r2']:.4f}")
        else:
            print(f"❌ Model training failed for {symbol}: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception during training: {e}")
    
    print("Model training test completed")



