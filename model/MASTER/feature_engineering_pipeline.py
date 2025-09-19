"""
Comprehensive Feature Engineering Pipeline for MASTER Model
Integrates market data, news sentiment, and trading schemes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from polygon_data_fetcher import PolygonDataFetcher
from news_sentiment_processor import NewsSentimentProcessor
from trading_schemes_features import TradingSchemesFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MASTERFeaturePipeline:
    """
    Comprehensive feature engineering pipeline for MASTER model
    """
    
    def __init__(self, polygon_api_key: str = None):
        """
        Initialize the feature engineering pipeline
        
        Args:
            polygon_api_key: Polygon.io API key
        """
        self.data_fetcher = PolygonDataFetcher(polygon_api_key)
        self.sentiment_processor = NewsSentimentProcessor()
        self.trading_schemes = TradingSchemesFeatures()
        
        # Market indices for broader context
        self.market_indices = [
            'SPY',   # S&P 500 ETF
            'QQQ',   # Nasdaq 100 ETF
            'IWM',   # Russell 2000 ETF
            'XLK',   # Technology sector
            'XLF',   # Financials sector
            'XLE',   # Energy sector
            'XLV',   # Healthcare sector
            'XLY',   # Consumer Discretionary sector
            'XLI'    # Industrials sector
        ]
    
    def fetch_market_data(self, tickers: List[str], from_date: str, to_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive market data for all tickers and indices
        
        Args:
            tickers: List of stock symbols
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with market data for each ticker/index
        """
        logger.info(f"Fetching market data for {len(tickers)} tickers from {from_date} to {to_date}")
        
        # Fetch stock data
        stock_data = {}
        for ticker in tickers:
            try:
                df = self.data_fetcher.get_technical_indicators(ticker, from_date, to_date)
                if not df.empty:
                    stock_data[ticker] = df
                    logger.info(f"Fetched data for {ticker}: {len(df)} records")
                else:
                    logger.warning(f"No data found for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        
        # Fetch market indices data
        index_data = self.data_fetcher.get_market_indices_data(self.market_indices, from_date, to_date)
        
        # Combine all data
        all_data = {**stock_data, **index_data}
        
        logger.info(f"Successfully fetched data for {len(all_data)} symbols")
        return all_data
    
    def fetch_news_sentiment(self, tickers: List[str], from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetch and process news sentiment for all tickers
        
        Args:
            tickers: List of stock symbols
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with aggregated sentiment data
        """
        logger.info(f"Fetching news sentiment for {len(tickers)} tickers")
        
        all_news = []
        
        for ticker in tickers:
            try:
                # Fetch news articles
                news_df = self.data_fetcher.get_news_sentiment(ticker, from_date, to_date, limit=100)
                
                if not news_df.empty:
                    # Process sentiment
                    processed_news = self.sentiment_processor.process_news_batch(news_df)
                    
                    # Filter for quality
                    filtered_news = self.sentiment_processor.filter_high_quality_news(
                        processed_news,
                        min_article_count=1
                    )
                    
                    if not filtered_news.empty:
                        all_news.append(filtered_news)
                        logger.info(f"Processed {len(filtered_news)} news articles for {ticker}")
                
            except Exception as e:
                logger.error(f"Error processing news for {ticker}: {e}")
        
        if all_news:
            combined_news = pd.concat(all_news, ignore_index=True)
            
            # Aggregate sentiment by time and ticker
            sentiment_features = self.sentiment_processor.get_market_sentiment_features(
                combined_news, tickers, from_date, to_date
            )
            
            logger.info(f"Generated sentiment features: {sentiment_features.shape}")
            return sentiment_features
        else:
            logger.warning("No news data found")
            return pd.DataFrame()
    
    def engineer_market_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Engineer market features including technical indicators and trading schemes
        
        Args:
            market_data: Dictionary with market data for each symbol
            
        Returns:
            Dictionary with engineered features for each symbol
        """
        logger.info("Engineering market features...")
        
        engineered_data = {}
        
        for symbol, data in market_data.items():
            try:
                # Calculate trading schemes
                trading_features = self.trading_schemes.calculate_all_trading_schemes(data)
                
                # Combine with original data
                combined_data = pd.concat([data, trading_features], axis=1)
                
                # Add market structure features
                combined_data = self._add_market_structure_features(combined_data)
                
                # Add volatility features
                combined_data = self._add_volatility_features(combined_data)
                
                # Add momentum features
                combined_data = self._add_momentum_features(combined_data)
                
                # Add liquidity features
                combined_data = self._add_liquidity_features(combined_data)
                
                engineered_data[symbol] = combined_data
                logger.info(f"Engineered features for {symbol}: {combined_data.shape[1]} features")
                
            except Exception as e:
                logger.error(f"Error engineering features for {symbol}: {e}")
        
        return engineered_data
    
    def _add_market_structure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features"""
        df = data.copy()
        
        # Price range
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Body size (close - open)
        df['body_size'] = (df['close'] - df['open']) / df['close']
        
        # Upper shadow
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        
        # Lower shadow
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        return df
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        df = data.copy()
        
        # Rolling volatility
        df['volatility_5'] = df['close'].pct_change().rolling(5).std()
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # High-low volatility
        df['hl_volatility'] = (df['high'] - df['low']) / df['close']
        
        # Volume-weighted volatility
        df['volume_weighted_volatility'] = df['volatility_20'] * df['volume']
        
        return df
    
    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        df = data.copy()
        
        # Price momentum
        df['momentum_1'] = df['close'].pct_change(1)
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(5)
        
        # VWAP momentum
        df['vwap_momentum'] = (df['close'] - df['VWAP']) / df['VWAP']
        
        return df
    
    def _add_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity features"""
        df = data.copy()
        
        # Volume features
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # Trade count features (if available)
        if 'trades' in df.columns:
            df['trades_ma_5'] = df['trades'].rolling(5).mean()
            df['trades_ratio'] = df['trades'] / df['trades_ma_5']
        
        # VWAP deviation
        df['vwap_deviation'] = (df['close'] - df['VWAP']) / df['VWAP']
        
        return df
    
    def create_multi_task_labels(self, engineered_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create multi-task labels for all symbols
        
        Args:
            engineered_data: Dictionary with engineered features
            
        Returns:
            Dictionary with labeled data for each symbol
        """
        logger.info("Creating multi-task labels...")
        
        labeled_data = {}
        
        for symbol, data in engineered_data.items():
            try:
                # Create trading scheme labels
                labeled_df = self.trading_schemes.create_trading_scheme_labels(data)
                
                # Add additional labels if needed
                labeled_df = self._add_additional_labels(labeled_df)
                
                labeled_data[symbol] = labeled_df
                logger.info(f"Created labels for {symbol}: {len([col for col in labeled_df.columns if col.endswith('_label')])} label types")
                
            except Exception as e:
                logger.error(f"Error creating labels for {symbol}: {e}")
        
        return labeled_data
    
    def _add_additional_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add additional labels for multi-task learning"""
        df = data.copy()
        
        # Risk-adjusted returns
        df['risk_adjusted_return'] = df['return_t+1'] / df['volatility_20']
        
        # Volatility labels
        df['volatility_label'] = df['volatility_20'].shift(-1)
        
        # Volume labels
        df['volume_label'] = df['volume_ratio'].shift(-1)
        
        return df
    
    def align_features_with_sentiment(self, engineered_data: Dict[str, pd.DataFrame], 
                                   sentiment_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Align market features with sentiment data
        
        Args:
            engineered_data: Dictionary with engineered market features
            sentiment_data: DataFrame with sentiment features
            
        Returns:
            Dictionary with aligned features
        """
        logger.info("Aligning features with sentiment data...")
        
        aligned_data = {}
        
        for symbol, data in engineered_data.items():
            try:
                # Get sentiment for this symbol
                symbol_sentiment = sentiment_data[sentiment_data['ticker'] == symbol].copy()
                
                if not symbol_sentiment.empty:
                    # Merge with market data
                    symbol_sentiment = symbol_sentiment.set_index('date')
                    data = data.set_index(data.index)  # Assuming index is datetime
                    
                    # Align by date
                    aligned_df = data.join(symbol_sentiment, how='left')
                    
                    # Fill missing sentiment with neutral values
                    sentiment_cols = ['sentiment_score', 'sentiment_std', 'article_count', 
                                   'positive_ratio', 'negative_ratio', 'neutral_ratio']
                    for col in sentiment_cols:
                        if col in aligned_df.columns:
                            aligned_df[col] = aligned_df[col].fillna(0.0 if 'score' in col else 0.33)
                    
                    aligned_data[symbol] = aligned_df
                    logger.info(f"Aligned features for {symbol}: {aligned_df.shape}")
                else:
                    # No sentiment data, use market data only
                    aligned_data[symbol] = data
                    logger.info(f"No sentiment data for {symbol}, using market data only")
                
            except Exception as e:
                logger.error(f"Error aligning features for {symbol}: {e}")
        
        return aligned_data
    
    def create_master_input_format(self, aligned_data: Dict[str, pd.DataFrame], 
                                 lookback_window: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input format for MASTER model
        
        Args:
            aligned_data: Dictionary with aligned features for each symbol
            lookback_window: Number of time steps to look back
            
        Returns:
            Tuple of (features, labels) in MASTER format
        """
        logger.info("Creating MASTER input format...")
        
        all_features = []
        all_labels = []
        
        for symbol, data in aligned_data.items():
            try:
                # Create feature matrix
                feature_matrix = self.trading_schemes.create_feature_matrix(data, lookback_window)
                
                # Create multi-task labels
                multi_task_labels = self.trading_schemes.prepare_multi_task_labels(data)
                
                # Align features with labels
                features, labels = self.trading_schemes.align_features_with_labels(
                    feature_matrix, multi_task_labels
                )
                
                if len(features) > 0 and len(labels) > 0:
                    all_features.append(features)
                    all_labels.append(labels)
                    logger.info(f"Created MASTER input for {symbol}: features {features.shape}, labels {labels.shape}")
                
            except Exception as e:
                logger.error(f"Error creating MASTER input for {symbol}: {e}")
        
        if all_features and all_labels:
            # Combine all symbols
            combined_features = np.concatenate(all_features, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)
            
            logger.info(f"Created MASTER input format: features {combined_features.shape}, labels {combined_labels.shape}")
            return combined_features, combined_labels
        else:
            logger.error("No valid data found for MASTER input")
            return np.array([]), np.array([])
    
    def run_full_pipeline(self, tickers: List[str], from_date: str, to_date: str,
                         lookback_window: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the complete feature engineering pipeline
        
        Args:
            tickers: List of stock symbols
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            lookback_window: Number of time steps to look back
            
        Returns:
            Tuple of (features, labels) ready for MASTER model
        """
        logger.info(f"Running full pipeline for {len(tickers)} tickers from {from_date} to {to_date}")
        
        # Step 1: Fetch market data
        market_data = self.fetch_market_data(tickers, from_date, to_date)
        
        # Step 2: Fetch news sentiment
        sentiment_data = self.fetch_news_sentiment(tickers, from_date, to_date)
        
        # Step 3: Engineer market features
        engineered_data = self.engineer_market_features(market_data)
        
        # Step 4: Create multi-task labels
        labeled_data = self.create_multi_task_labels(engineered_data)
        
        # Step 5: Align features with sentiment
        aligned_data = self.align_features_with_sentiment(labeled_data, sentiment_data)
        
        # Step 6: Create MASTER input format
        features, labels = self.create_master_input_format(aligned_data, lookback_window)
        
        logger.info("Pipeline completed successfully!")
        return features, labels
    
    def validate_feature_dimensions(self, features: np.ndarray, expected_dim: int) -> bool:
        """
        Validate feature dimensions to ensure consistency
        
        Args:
            features: Feature array
            expected_dim: Expected feature dimension
            
        Returns:
            True if dimensions match, False otherwise
        """
        if len(features.shape) != 3:
            logger.error(f"Features must be 3D array (N, T, F), got shape: {features.shape}")
            return False
        
        actual_dim = features.shape[-1]
        if actual_dim != expected_dim:
            logger.error(f"Feature dimension mismatch: expected {expected_dim}, got {actual_dim}")
            return False
        
        logger.info(f"Feature dimensions validated: {features.shape}")
        return True
    
    def align_time_series(self, market_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align time series data to ensure consistent timestamps
        
        Args:
            market_data: Market data DataFrame
            sentiment_data: Sentiment data DataFrame
            
        Returns:
            Tuple of aligned DataFrames
        """
        if sentiment_data.empty:
            logger.warning("No sentiment data to align")
            return market_data, pd.DataFrame()
        
        # Ensure both datasets have datetime index
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)
        
        if not isinstance(sentiment_data.index, pd.DatetimeIndex):
            sentiment_data.index = pd.to_datetime(sentiment_data.index)
        
        # Find common time index
        common_index = market_data.index.intersection(sentiment_data.index)
        
        if len(common_index) == 0:
            logger.warning("No common timestamps between market and sentiment data")
            return market_data, pd.DataFrame()
        
        # Align data to common index
        aligned_market = market_data.loc[common_index]
        aligned_sentiment = sentiment_data.loc[common_index]
        
        logger.info(f"Aligned {len(common_index)} common timestamps")
        return aligned_market, aligned_sentiment


# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MASTERFeaturePipeline()
    
    # Example tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    from_date = '2024-01-01'
    to_date = '2024-01-31'
    
    # Run full pipeline
    print("Running full feature engineering pipeline...")
    features, labels = pipeline.run_full_pipeline(tickers, from_date, to_date)
    
    if len(features) > 0:
        print(f"Pipeline completed successfully!")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of tasks: {labels.shape[1] if len(labels.shape) > 1 else 1}")
    else:
        print("Pipeline failed - no data generated")
