"""
Feature Engineering for LSTM Stock Prediction
Based on JordiCorbilla/stock-prediction-deep-neural-learning approach
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering for stock prediction following JordiCorbilla approach
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        # Use only close price to match the current trained model
        # TODO: Retrain model with 6 features for better predictions
        self.features = ['close']
        logger.info("FeatureEngineer initialized with close price feature (matching current trained model)")
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features to the dataframe - using close price only for current model
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with close price feature only (matching current trained model)
        """
        try:
            if df.empty:
                return df
            
            # Create a copy to avoid modifying original
            data = df.copy()
            
            # Ensure we have the close price
            if 'close' not in data.columns:
                logger.error("Close price column not found in data")
                return df
            
            # Select only the close price column
            result = data[['close']].copy()
            
            # Fill any NaN values
            result = result.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure values are numeric
            result['close'] = pd.to_numeric(result['close'], errors='coerce')
            
            # Fill any remaining NaN values with 0
            result = result.fillna(0)
            
            logger.info(f"Using close price feature only (matching current trained model)")
            return result
            
        except Exception as e:
            logger.error(f"Error adding features: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.features
    
    def prepare_prediction_data(self, df: pd.DataFrame, sequence_length: int) -> Optional[np.ndarray]:
        """
        Prepare data for prediction following JordiCorbilla approach
        
        Args:
            df: DataFrame with features
            sequence_length: Number of time steps for sequence
            
        Returns:
            Prepared sequence array or None if failed
        """
        try:
            if df.empty or len(df) < sequence_length:
                logger.error(f"Insufficient data: need {sequence_length}, got {len(df)}")
                return None
            
            # Select only the features we need
            feature_data = df[self.features].tail(sequence_length)
            
            # Convert to numpy array
            sequence = feature_data.values.reshape(1, sequence_length, len(self.features))
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            return None
