"""
Trading Schemes Feature Engineering for MASTER Model
Implements weighted trading scheme signals as features and labels
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSchemesFeatures:
    """
    Generates weighted trading scheme features for MASTER model
    """
    
    def __init__(self):
        """Initialize the trading schemes feature generator"""
        pass
    
    def calculate_sma_signals(self, data: pd.DataFrame, short_window: int = 5, 
                            long_window: int = 20) -> pd.DataFrame:
        """
        Calculate SMA crossover signals
        
        Args:
            data: DataFrame with OHLCV data
            short_window: Short SMA window
            long_window: Long SMA window
            
        Returns:
            DataFrame with SMA signals
        """
        df = data.copy()
        
        # Calculate SMAs
        df['SMA_short'] = df['close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=long_window).mean()
        
        # Calculate weighted signal
        df['SMA_signal_weighted'] = (df['SMA_short'] - df['SMA_long']) / df['SMA_long']
        df['SMA_signal_weighted'] = df['SMA_signal_weighted'].clip(-1, 1)
        
        return df[['SMA_short', 'SMA_long', 'SMA_signal_weighted']]
    
    def calculate_ema_signals(self, data: pd.DataFrame, short_span: int = 5, 
                            long_span: int = 20) -> pd.DataFrame:
        """
        Calculate EMA crossover signals
        
        Args:
            data: DataFrame with OHLCV data
            short_span: Short EMA span
            long_span: Long EMA span
            
        Returns:
            DataFrame with EMA signals
        """
        df = data.copy()
        
        # Calculate EMAs
        df['EMA_short'] = df['close'].ewm(span=short_span, adjust=False).mean()
        df['EMA_long'] = df['close'].ewm(span=long_span, adjust=False).mean()
        
        # Calculate weighted signal
        df['EMA_signal_weighted'] = (df['EMA_short'] - df['EMA_long']) / df['EMA_long']
        df['EMA_signal_weighted'] = df['EMA_signal_weighted'].clip(-1, 1)
        
        return df[['EMA_short', 'EMA_long', 'EMA_signal_weighted']]
    
    def calculate_macd_signals(self, data: pd.DataFrame, fast: int = 12, 
                             slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD histogram signals
        
        Args:
            data: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD signals
        """
        df = data.copy()
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        df['MACD_hist'] = macd_line - signal_line
        
        # Calculate weighted signal (normalized by rolling standard deviation)
        macd_std = df['MACD_hist'].rolling(window=20).std()
        df['MACD_signal_weighted'] = df['MACD_hist'] / macd_std
        df['MACD_signal_weighted'] = df['MACD_signal_weighted'].clip(-1, 1)
        
        return df[['MACD_hist', 'MACD_signal_weighted']]
    
    def calculate_rsi_signals(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Calculate RSI threshold signals
        
        Args:
            data: DataFrame with OHLCV data
            window: RSI calculation window
            
        Returns:
            DataFrame with RSI signals
        """
        df = data.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate weighted signal based on distance from 50
        df['RSI_signal_weighted'] = 0
        df.loc[df['RSI'] < 50, 'RSI_signal_weighted'] = (50 - df['RSI']) / 50
        df.loc[df['RSI'] > 50, 'RSI_signal_weighted'] = (50 - df['RSI']) / 50
        df['RSI_signal_weighted'] = df['RSI_signal_weighted'].clip(-1, 1)
        
        return df[['RSI', 'RSI_signal_weighted']]
    
    def calculate_bollinger_bands_signals(self, data: pd.DataFrame, window: int = 20, 
                                       num_std: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Band breakout signals
        
        Args:
            data: DataFrame with OHLCV data
            window: Rolling window for Bollinger Bands
            num_std: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Band signals
        """
        df = data.copy()
        
        # Calculate Bollinger Bands
        df['BB_mid'] = df['close'].rolling(window=window).mean()
        df['BB_std'] = df['close'].rolling(window=window).std()
        df['BB_upper'] = df['BB_mid'] + num_std * df['BB_std']
        df['BB_lower'] = df['BB_mid'] - num_std * df['BB_std']
        
        # Calculate weighted signal for breakouts
        df['BB_signal_weighted'] = 0
        
        # Upper band breakout (bearish)
        upper_breakout = df['close'] > df['BB_upper']
        df.loc[upper_breakout, 'BB_signal_weighted'] = (df['close'] - df['BB_upper']) / df['BB_upper']
        
        # Lower band breakout (bullish)
        lower_breakout = df['close'] < df['BB_lower']
        df.loc[lower_breakout, 'BB_signal_weighted'] = (df['close'] - df['BB_lower']) / df['BB_lower']
        
        df['BB_signal_weighted'] = df['BB_signal_weighted'].clip(-1, 1)
        
        return df[['BB_upper', 'BB_mid', 'BB_lower', 'BB_signal_weighted']]
    
    def calculate_volume_signals(self, data: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Calculate volume spike signals
        
        Args:
            data: DataFrame with OHLCV data
            window: Rolling window for volume average
            
        Returns:
            DataFrame with volume signals
        """
        df = data.copy()
        
        # Calculate volume average
        df['vol_avg'] = df['volume'].rolling(window=window).mean()
        
        # Calculate weighted signal for volume spikes
        df['Volume_signal_weighted'] = (df['volume'] - df['vol_avg']) / df['vol_avg']
        df['Volume_signal_weighted'] = df['Volume_signal_weighted'].clip(0, 3)  # Only positive spikes
        
        return df[['vol_avg', 'Volume_signal_weighted']]
    
    def calculate_vwap_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP trend signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VWAP signals
        """
        df = data.copy()
        
        # Calculate VWAP with zero volume handling
        volume_cumsum = df['volume'].cumsum()
        price_volume_cumsum = (df['close'] * df['volume']).cumsum()
        
        df['VWAP'] = np.where(volume_cumsum > 0, 
                             price_volume_cumsum / volume_cumsum,
                             df['close'])
        
        # Calculate weighted signal based on distance from VWAP
        df['VWAP_signal_weighted'] = (df['close'] - df['VWAP']) / df['VWAP']
        df['VWAP_signal_weighted'] = df['VWAP_signal_weighted'].clip(-1, 1)
        
        return df[['VWAP', 'VWAP_signal_weighted']]
    
    def calculate_all_trading_schemes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all trading scheme signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all trading scheme signals
        """
        df = data.copy()
        
        # Calculate all signals
        sma_signals = self.calculate_sma_signals(df)
        ema_signals = self.calculate_ema_signals(df)
        macd_signals = self.calculate_macd_signals(df)
        rsi_signals = self.calculate_rsi_signals(df)
        bb_signals = self.calculate_bollinger_bands_signals(df)
        volume_signals = self.calculate_volume_signals(df)
        vwap_signals = self.calculate_vwap_signals(df)
        
        # Combine all signals
        all_signals = pd.concat([
            sma_signals,
            ema_signals,
            macd_signals,
            rsi_signals,
            bb_signals,
            volume_signals,
            vwap_signals
        ], axis=1)
        
        return all_signals
    
    def create_trading_scheme_labels(self, data: pd.DataFrame, 
                                   return_column: str = 'close') -> pd.DataFrame:
        """
        Create future labels for trading schemes (shifted by 1 time step)
        
        Args:
            data: DataFrame with trading scheme signals
            return_column: Column to use for return calculation
            
        Returns:
            DataFrame with future labels
        """
        df = data.copy()
        
        # Calculate future returns
        df['return_t+1'] = df[return_column].shift(-1) / df[return_column] - 1
        
        # Create directional labels
        threshold = 0.001  # 0.1% move
        df['direction_t+1'] = 0
        df.loc[df['return_t+1'] > threshold, 'direction_t+1'] = 1
        df.loc[df['return_t+1'] < -threshold, 'direction_t+1'] = -1
        
        # Create future trading scheme labels (shifted by -1)
        signal_columns = [col for col in df.columns if col.endswith('_signal_weighted')]
        
        for col in signal_columns:
            label_col = col.replace('_signal_weighted', '_label')
            df[label_col] = df[col].shift(-1)
        
        return df
    
    def prepare_multi_task_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare multi-task labels for MASTER model
        
        Args:
            data: DataFrame with all features and labels
            
        Returns:
            Array of shape (N, T, num_tasks) with multi-task labels
        """
        # Get primary return label
        return_label = data['return_t+1'].values
        
        # Get trading scheme labels
        label_columns = [col for col in data.columns if col.endswith('_label')]
        scheme_labels = data[label_columns].values
        
        # Combine all labels
        all_labels = np.column_stack([return_label] + [scheme_labels[:, i] for i in range(scheme_labels.shape[1])])
        
        return all_labels
    
    def create_feature_matrix(self, data: pd.DataFrame, lookback_window: int = 8) -> np.ndarray:
        """
        Create feature matrix for MASTER model
        
        Args:
            data: DataFrame with all features
            lookback_window: Number of time steps to look back
            
        Returns:
            Array of shape (N, T, F) with features
        """
        # Select feature columns (exclude labels and raw OHLCV)
        feature_columns = [col for col in data.columns 
                          if not col.endswith('_label') and 
                          not col in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']]
        
        features = data[feature_columns].values
        
        # Create sliding windows
        n_samples, n_features = features.shape
        feature_matrix = []
        
        for i in range(lookback_window, n_samples):
            window = features[i-lookback_window:i]
            feature_matrix.append(window)
        
        return np.array(feature_matrix)
    
    def align_features_with_labels(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align features with labels to prevent future leakage
        
        Args:
            features: Feature matrix (N, T, F)
            labels: Label matrix (N, num_tasks)
            
        Returns:
            Tuple of aligned features and labels
        """
        # Ensure same length
        min_length = min(len(features), len(labels))
        features = features[:min_length]
        labels = labels[:min_length]
        
        return features, labels


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.01) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.01) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.01),
        'volume': np.random.randint(1000, 10000, 100),
        'vwap': 100 + np.cumsum(np.random.randn(100) * 0.01)
    })
    
    # Initialize trading schemes
    schemes = TradingSchemesFeatures()
    
    # Calculate all trading schemes
    print("Calculating trading schemes...")
    all_signals = schemes.calculate_all_trading_schemes(sample_data)
    print(f"Generated {all_signals.shape[1]} trading scheme features")
    
    # Create labels
    print("\nCreating labels...")
    labeled_data = schemes.create_trading_scheme_labels(sample_data)
    print(f"Created labels: {[col for col in labeled_data.columns if col.endswith('_label')]}")
    
    # Prepare multi-task labels
    print("\nPreparing multi-task labels...")
    multi_task_labels = schemes.prepare_multi_task_labels(labeled_data)
    print(f"Multi-task labels shape: {multi_task_labels.shape}")
    
    # Create feature matrix
    print("\nCreating feature matrix...")
    feature_matrix = schemes.create_feature_matrix(labeled_data, lookback_window=8)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    print("\nTrading schemes features created successfully!")
