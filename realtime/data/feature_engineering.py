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
        # Use all available fields for comprehensive predictions
        # OHLCV + transactions + VWAP + technical indicators
        self.features = [
            'open', 'high', 'low', 'close', 'volume', 
            'transactions', 'vwap',
            'rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower',
            'price_change', 'volume_change', 'price_volume_ratio'
        ]
        logger.info("FeatureEngineer initialized with comprehensive features for enhanced predictions")
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive features to the dataframe using all available data
        
        Args:
            df: DataFrame with OHLCV + transactions + VWAP data
            
        Returns:
            DataFrame with comprehensive features for enhanced predictions
        """
        try:
            if df.empty:
                return df
            
            # Create a copy to avoid modifying original
            data = df.copy()
            
            # Ensure we have the required base columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return df
            
            # Initialize result with base OHLCV data
            result = data[required_columns].copy()
            
            # Add transactions and VWAP if available
            if 'transactions' in data.columns:
                result['transactions'] = pd.to_numeric(data['transactions'], errors='coerce')
            else:
                result['transactions'] = 0
                
            if 'vwap' in data.columns:
                result['vwap'] = pd.to_numeric(data['vwap'], errors='coerce')
            else:
                result['vwap'] = result['close']  # Use close price as fallback
            
            # Calculate technical indicators
            result['rsi'] = self._calculate_rsi(result['close'])
            result['macd'] = self._calculate_macd(result['close'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(result['close'])
            result['bb_upper'] = bb_upper
            result['bb_middle'] = bb_middle
            result['bb_lower'] = bb_lower
            
            # Price and volume change features
            result['price_change'] = result['close'].pct_change()
            result['volume_change'] = result['volume'].pct_change()
            result['price_volume_ratio'] = result['close'] / (result['volume'] + 1)  # +1 to avoid division by zero
            
            # Fill any NaN values using forward fill then backward fill
            result = result.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure all values are numeric
            for col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')
            
            # Fill any remaining NaN values with 0
            result = result.fillna(0)
            
            logger.info(f"Created comprehensive feature set with {len(result.columns)} features")
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
