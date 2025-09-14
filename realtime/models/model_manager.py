"""
Model Management System for Real-time LSTM Prediction System
Phase 1.1.1: Model Loader System with Validation & Health Checks
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages pre-trained LSTM models for real-time prediction
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model manager
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = models_dir
        self.loaded_models = {}  # Cache for loaded models
        self.model_configs = {}  # Cache for model configurations
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info(f"ModelManager initialized with models directory: {models_dir}")
    
    def _create_default_config(self, stock_symbol: str, model_path: str, model: tf.keras.Model = None) -> Dict[str, Any]:
        """
        Create a default configuration by analyzing the model and scaler
        
        Args:
            stock_symbol: Stock symbol
            model_path: Path to model directory
            model: Pre-loaded model (optional)
            
        Returns:
            Default configuration dictionary
        """
        try:
            # Use provided model or load it
            if model is None:
                # Find model file
                model_weights_h5 = os.path.join(model_path, "model_weights.h5")
                model_weights_keras = os.path.join(model_path, "model_weights.keras")
                
                model_file = None
                if os.path.exists(model_weights_h5):
                    model_file = model_weights_h5
                elif os.path.exists(model_weights_keras):
                    model_file = model_weights_keras
                
                if not model_file:
                    raise ValueError("No model file found")
                
                # Load model to analyze architecture
                model = tf.keras.models.load_model(model_file)
            
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            
            # Extract sequence length and number of features
            if len(input_shape) == 3:  # (batch, sequence_length, features)
                sequence_length = input_shape[1]
                num_features = input_shape[2]
            else:
                # Fallback values
                sequence_length = 60
                num_features = 5
            
            # Load scaler to understand feature structure
            scaler_path = os.path.join(model_path, "scaler.pkl")
            scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            
            # Create default features list
            default_features = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_5', 'sma_10', 'sma_20',
                'ema_5', 'ema_10', 'ema_20',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower',
                'atr', 'stoch_k', 'stoch_d',
                'price_change', 'volume_change', 'volatility'
            ]
            
            # Use the number of features from the model
            features = default_features[:num_features]
            if num_features > len(default_features):
                for i in range(len(default_features), num_features):
                    features.append(f'feature_{i}')
            
            # Create default configuration
            config = {
                "symbol": stock_symbol,
                "model_type": "LSTM",
                "sequence_length": sequence_length,
                "prediction_length": 672,  # 1 week = 7 days * 24 hours * 4 (15-min intervals)
                "features": features,
                "model_architecture": {
                    "input_shape": input_shape,
                    "total_params": model.count_params(),
                    "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
                    "non_trainable_params": sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
                },
                "scaling": {
                    "scaler_type": type(scaler).__name__ if scaler else "Unknown",
                    "feature_range": getattr(scaler, 'feature_range', None) if scaler else None
                },
                "training_info": {
                    "created_at": datetime.now().isoformat(),
                    "data_source": "Unknown",
                    "timeframe": "15min",
                    "lookback_days": 30
                },
                "performance_metrics": {
                    "rmse": 0.0,
                    "mae": 0.0,
                    "r2": 0.0,
                    "accuracy": 0.0
                },
                "prediction_settings": {
                    "confidence_threshold": 0.6,
                    "validation_threshold": 0.02,
                    "min_profit_percent": 2.0
                },
                "auto_generated": True
            }
            
            logger.info(f"Created default config for {stock_symbol}: sequence_length={sequence_length}, features={len(features)}")
            return config
            
        except Exception as e:
            logger.error(f"Error creating default config for {stock_symbol}: {e}")
            # Return minimal config as fallback
            return {
                "symbol": stock_symbol,
                "model_type": "LSTM",
                "sequence_length": 60,
                "prediction_length": 672,
                "features": ["open", "high", "low", "close", "volume"],
                "auto_generated": True
            }
    
    def load_model(self, stock_symbol: str, model_path: str = None) -> bool:
        """
        Load a pre-trained model for a stock symbol
        
        Args:
            stock_symbol: Stock symbol (e.g., 'AAPL')
            model_path: Path to model directory (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_path is None:
                model_path = os.path.join(self.models_dir, stock_symbol)
            
            if not os.path.exists(model_path):
                logger.error(f"Model path does not exist: {model_path}")
                return False
            
            # Find model weights file (try both .h5 and .keras formats)
            model_weights_h5 = os.path.join(model_path, "model_weights.h5")
            model_weights_keras = os.path.join(model_path, "model_weights.keras")
            
            model_weights_path = None
            if os.path.exists(model_weights_h5):
                model_weights_path = model_weights_h5
            elif os.path.exists(model_weights_keras):
                model_weights_path = model_weights_keras
            else:
                logger.error(f"Model weights not found. Expected either {model_weights_h5} or {model_weights_keras}")
                return False
            
            # Load model weights
            model = tf.keras.models.load_model(model_weights_path)
            
            # Try to load model configuration, create default if not found
            config_path = os.path.join(model_path, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded existing config for {stock_symbol}")
            else:
                logger.info(f"No config file found for {stock_symbol}, creating default config")
                config = self._create_default_config(stock_symbol, model_path, model)
            
            # Load scaler
            scaler_path = os.path.join(model_path, "scaler.pkl")
            if not os.path.exists(scaler_path):
                logger.error(f"Scaler not found: {scaler_path}")
                return False
            
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Store in cache
            self.loaded_models[stock_symbol] = {
                'model': model,
                'scaler': scaler,
                'config': config,
                'loaded_at': datetime.now()
            }
            
            self.model_configs[stock_symbol] = config
            
            logger.info(f"Successfully loaded model for {stock_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {stock_symbol}: {e}")
            return False
    
    def get_model(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a loaded model
        
        Args:
            stock_symbol: Stock symbol
            
        Returns:
            Dictionary with model, scaler, and config, or None if not loaded
        """
        if stock_symbol not in self.loaded_models:
            logger.warning(f"Model for {stock_symbol} not loaded")
            return None
        
        return self.loaded_models[stock_symbol]
    
    def list_available_models(self) -> List[str]:
        """
        List all available model directories
        
        Returns:
            List of stock symbols with available models
        """
        try:
            available_models = []
            
            if not os.path.exists(self.models_dir):
                return available_models
            
            for item in os.listdir(self.models_dir):
                model_path = os.path.join(self.models_dir, item)
                if os.path.isdir(model_path):
                    # Check if all required files exist (model can be .h5 or .keras)
                    required_files = ["scaler.pkl"]
                    has_model = (os.path.exists(os.path.join(model_path, "model_weights.h5")) or 
                               os.path.exists(os.path.join(model_path, "model_weights.keras")))
                    if has_model and all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                        available_models.append(item)
            
            logger.info(f"Found {len(available_models)} available models: {available_models}")
            return available_models
            
        except Exception as e:
            logger.error(f"Error listing available models: {e}")
            return []
    
    def unload_model(self, stock_symbol: str) -> bool:
        """
        Unload a model from memory
        
        Args:
            stock_symbol: Stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if stock_symbol in self.loaded_models:
                del self.loaded_models[stock_symbol]
                logger.info(f"Unloaded model for {stock_symbol}")
                return True
            else:
                logger.warning(f"Model for {stock_symbol} not loaded")
                return False
                
        except Exception as e:
            logger.error(f"Error unloading model for {stock_symbol}: {e}")
            return False
    
    def validate_model_loading(self, model_path: str) -> bool:
        """
        Validate that a model can be loaded without errors
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if all required files exist (model can be .h5 or .keras)
            required_files = ["scaler.pkl"]
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    logger.error(f"Required file missing: {file_path}")
                    return False
            
            # Check for model file
            model_weights_h5 = os.path.join(model_path, "model_weights.h5")
            model_weights_keras = os.path.join(model_path, "model_weights.keras")
            
            if not os.path.exists(model_weights_h5) and not os.path.exists(model_weights_keras):
                logger.error(f"Model weights not found. Expected either {model_weights_h5} or {model_weights_keras}")
                return False
            
            # Try to load the model
            model_weights_path = model_weights_h5 if os.path.exists(model_weights_h5) else model_weights_keras
            model = tf.keras.models.load_model(model_weights_path)
            
            # Try to load the scaler
            scaler_path = os.path.join(model_path, "scaler.pkl")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Try to load the config, create default if not found
            config_path = os.path.join(model_path, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Create default config for validation using the already loaded model
                config = self._create_default_config(os.path.basename(model_path), model_path, model)
            
            logger.info(f"Model validation successful for {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed for {model_path}: {e}")
            return False
    
    def check_model_compatibility(self, model: tf.keras.Model, data_shape: Tuple[int, int]) -> bool:
        """
        Check if model is compatible with input data shape
        
        Args:
            model: TensorFlow model
            data_shape: Expected input data shape (sequence_length, features)
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Get model input shape
            model_input_shape = model.input_shape
            if isinstance(model_input_shape, list):
                model_input_shape = model_input_shape[0]
            
            # Check if shapes match (excluding batch dimension)
            if len(model_input_shape) != 3:  # Should be (batch, sequence, features)
                logger.error(f"Model input shape should be 3D, got {len(model_input_shape)}D")
                return False
            
            expected_shape = (None, data_shape[0], data_shape[1])
            if model_input_shape[1:] != expected_shape[1:]:
                logger.error(f"Shape mismatch: model expects {model_input_shape[1:]}, data has {data_shape}")
                return False
            
            logger.info(f"Model compatibility check passed for shape {data_shape}")
            return True
            
        except Exception as e:
            logger.error(f"Model compatibility check failed: {e}")
            return False
    
    def validate_prediction_output(self, prediction: np.ndarray) -> bool:
        """
        Validate prediction output
        
        Args:
            prediction: Model prediction output
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if prediction is not NaN or infinite
            if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
                logger.error("Prediction contains NaN or infinite values")
                return False
            
            # Check if prediction is reasonable (positive price)
            if np.any(prediction <= 0):
                logger.warning("Prediction contains non-positive values")
                return False
            
            # Check if prediction is within reasonable bounds (not too extreme)
            if np.any(prediction > 10000) or np.any(prediction < 0.01):
                logger.warning("Prediction contains extreme values")
                return False
            
            logger.debug("Prediction output validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Prediction output validation failed: {e}")
            return False
    
    def run_health_check(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Run comprehensive health check on a model
        
        Args:
            stock_symbol: Stock symbol
            
        Returns:
            Dictionary with health check results
        """
        try:
            health_status = {
                'symbol': stock_symbol,
                'timestamp': datetime.now(),
                'overall_status': 'healthy',
                'checks': {}
            }
            
            # Check if model is loaded
            if stock_symbol not in self.loaded_models:
                health_status['overall_status'] = 'unhealthy'
                health_status['checks']['model_loaded'] = {
                    'status': 'failed',
                    'message': 'Model not loaded'
                }
                return health_status
            
            model_data = self.loaded_models[stock_symbol]
            model = model_data['model']
            scaler = model_data['scaler']
            config = model_data['config']
            
            # Check model loading
            health_status['checks']['model_loaded'] = {
                'status': 'passed',
                'message': 'Model successfully loaded'
            }
            
            # Check model configuration
            required_config_keys = ['sequence_length', 'prediction_length', 'features']
            missing_keys = [key for key in required_config_keys if key not in config]
            
            if missing_keys:
                health_status['checks']['config_complete'] = {
                    'status': 'failed',
                    'message': f'Missing config keys: {missing_keys}'
                }
            else:
                health_status['checks']['config_complete'] = {
                    'status': 'passed',
                    'message': 'Configuration complete'
                }
            
            # Test prediction with sample data
            try:
                sequence_length = config['sequence_length']
                features = config['features']
                
                # Create sample data with realistic price values
                # Generate data that looks like stock prices (positive values)
                sample_data = np.random.uniform(50, 200, (1, sequence_length, len(features)))
                sample_data = scaler.transform(sample_data.reshape(-1, len(features)))
                sample_data = sample_data.reshape(1, sequence_length, len(features))
                
                # Make prediction
                prediction = model.predict(sample_data, verbose=0)
                
                # Validate prediction
                if self.validate_prediction_output(prediction):
                    health_status['checks']['prediction_test'] = {
                        'status': 'passed',
                        'message': 'Prediction test successful'
                    }
                else:
                    health_status['checks']['prediction_test'] = {
                        'status': 'failed',
                        'message': 'Prediction output validation failed'
                    }
                    health_status['overall_status'] = 'unhealthy'
                
            except Exception as e:
                health_status['checks']['prediction_test'] = {
                    'status': 'failed',
                    'message': f'Prediction test failed: {str(e)}'
                }
                health_status['overall_status'] = 'unhealthy'
            
            # Check model performance metrics
            if 'performance_metrics' in config:
                metrics = config['performance_metrics']
                health_status['checks']['performance_metrics'] = {
                    'status': 'passed',
                    'message': f'RMSE: {metrics.get("rmse", "N/A")}, R²: {metrics.get("r2", "N/A")}'
                }
            else:
                health_status['checks']['performance_metrics'] = {
                    'status': 'warning',
                    'message': 'No performance metrics available'
                }
            
            logger.info(f"Health check completed for {stock_symbol}: {health_status['overall_status']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed for {stock_symbol}: {e}")
            return {
                'symbol': stock_symbol,
                'timestamp': datetime.now(),
                'overall_status': 'unhealthy',
                'checks': {
                    'health_check': {
                        'status': 'failed',
                        'message': f'Health check failed: {str(e)}'
                    }
                }
            }
    
    def get_model_info(self, stock_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model
        
        Args:
            stock_symbol: Stock symbol
            
        Returns:
            Dictionary with model information, or None if not found
        """
        try:
            if stock_symbol not in self.loaded_models:
                return None
            
            model_data = self.loaded_models[stock_symbol]
            config = model_data['config']
            model = model_data['model']
            
            info = {
                'symbol': stock_symbol,
                'loaded_at': model_data['loaded_at'],
                'config': config,
                'model_summary': {
                    'total_params': model.count_params(),
                    'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
                    'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
                }
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info for {stock_symbol}: {e}")
            return None
    
    def cleanup_old_models(self, max_age_hours: int = 24) -> int:
        """
        Clean up models that haven't been used recently
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of models cleaned up
        """
        try:
            current_time = datetime.now()
            models_to_remove = []
            
            for symbol, model_data in self.loaded_models.items():
                age_hours = (current_time - model_data['loaded_at']).total_seconds() / 3600
                if age_hours > max_age_hours:
                    models_to_remove.append(symbol)
            
            for symbol in models_to_remove:
                self.unload_model(symbol)
            
            logger.info(f"Cleaned up {len(models_to_remove)} old models")
            return len(models_to_remove)
            
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
            return 0


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the model manager
    manager = ModelManager("test_models")
    
    # List available models
    available_models = manager.list_available_models()
    print(f"Available models: {available_models}")
    
    # Test loading a model (if available)
    if available_models:
        symbol = available_models[0]
        success = manager.load_model(symbol)
        print(f"Model loading for {symbol}: {'Success' if success else 'Failed'}")
        
        if success:
            # Run health check
            health = manager.run_health_check(symbol)
            print(f"Health check for {symbol}: {health['overall_status']}")
            
            # Get model info
            info = manager.get_model_info(symbol)
            if info:
                print(f"Model info: {info['model_summary']}")
    
    print("Model manager test completed")
