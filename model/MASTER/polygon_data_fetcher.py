"""
Polygon.io Data Fetcher for MASTER Model Integration
Fetches market data, technical indicators, and news sentiment data from Polygon.io
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import logging
import warnings
from scipy.stats import spearmanr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolygonDataFetcher:
    """
    Fetches market data, technical indicators, and news data from Polygon.io
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Polygon data fetcher
        
        Args:
            api_key: Polygon.io API key. If None, will try to load from .env file
        """
        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('POLYGON_API_KEY')
            
        if not api_key:
            raise ValueError("Polygon API key not found. Please set POLYGON_API_KEY in .env file")
            
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make a request to Polygon.io API with rate limiting
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response data
        """
        if params is None:
            params = {}
            
        params['apikey'] = self.api_key
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Rate limiting - Polygon allows 5 requests per minute for free tier
            time.sleep(0.2)  # 200ms delay between requests
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            raise
    
    def get_aggregates(self, ticker: str, from_date: str, to_date: str, 
                      timespan: str = "day", multiplier: int = 1) -> pd.DataFrame:
        """
        Get OHLCV aggregates data for a ticker
        
        Args:
            ticker: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timespan: day, hour, minute
            multiplier: Size of the timespan multiplier
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        data = self._make_request(endpoint)
        
        if 'results' not in data:
            logger.warning(f"No data found for {ticker}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('timestamp')
        
        # Rename columns to standard OHLCV format
        df = df.rename(columns={
            'o': 'open',
            'h': 'high', 
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'trades'
        })
        
        return df[['open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']]
    
    def get_technical_indicators(self, ticker: str, from_date: str, to_date: str,
                               indicators: List[str] = None) -> pd.DataFrame:
        """
        Get technical indicators for a ticker
        
        Args:
            ticker: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            indicators: List of indicators to fetch
            
        Returns:
            DataFrame with technical indicators
        """
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd', 'bbands']
            
        endpoint = f"/v1/indicators/{indicators[0]}/{ticker}"
        params = {
            'timestamp.gte': from_date,
            'timestamp.lte': to_date,
            'adjusted': 'true'
        }
        
        # For now, we'll implement basic technical indicators
        # Polygon's technical indicators API has limited free tier access
        # We'll calculate indicators from OHLCV data instead
        ohlcv_data = self.get_aggregates(ticker, from_date, to_date)
        
        if ohlcv_data.empty:
            return pd.DataFrame()
            
        # Calculate technical indicators
        df = ohlcv_data.copy()
        
        # Simple Moving Average
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        macd_data = self._calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        df['bb_width'] = bb_data['width']
        df['bb_position'] = bb_data['position']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        width = (upper - lower) / middle
        position = (prices - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'position': position
        }
    
    def get_market_snapshots(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get current market snapshots for multiple tickers
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            DataFrame with current market data
        """
        endpoint = "/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {
            'tickers': ','.join(tickers)
        }
        
        data = self._make_request(endpoint, params)
        
        if 'tickers' not in data:
            return pd.DataFrame()
            
        results = []
        for ticker_data in data['tickers']:
            if 'day' in ticker_data and 'prevDay' in ticker_data:
                day_data = ticker_data['day']
                prev_data = ticker_data['prevDay']
                
                results.append({
                    'ticker': ticker_data['ticker'],
                    'close': day_data['c'],
                    'high': day_data['h'],
                    'low': day_data['l'],
                    'open': day_data['o'],
                    'volume': day_data['v'],
                    'vwap': day_data['vw'],
                    'change': day_data['c'] - prev_data['c'],
                    'change_percent': (day_data['c'] - prev_data['c']) / prev_data['c'] * 100
                })
        
        return pd.DataFrame(results)
    
    def get_market_gainers_losers(self, limit: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get top gainers and losers for market breadth signals
        
        Args:
            limit: Number of top gainers/losers to fetch
            
        Returns:
            Tuple of (gainers_df, losers_df)
        """
        # Get gainers
        gainers_endpoint = "/v2/snapshot/locale/us/markets/stocks/gainers"
        gainers_data = self._make_request(gainers_endpoint, {'limit': limit})
        
        # Get losers  
        losers_endpoint = "/v2/snapshot/locale/us/markets/stocks/losers"
        losers_data = self._make_request(losers_endpoint, {'limit': limit})
        
        gainers_df = pd.DataFrame()
        losers_df = pd.DataFrame()
        
        if 'tickers' in gainers_data:
            gainers_list = []
            for ticker_data in gainers_data['tickers']:
                if 'day' in ticker_data and 'prevDay' in ticker_data:
                    day_data = ticker_data['day']
                    prev_data = ticker_data['prevDay']
                    
                    gainers_list.append({
                        'ticker': ticker_data['ticker'],
                        'change_percent': (day_data['c'] - prev_data['c']) / prev_data['c'] * 100,
                        'volume': day_data['v']
                    })
            gainers_df = pd.DataFrame(gainers_list)
        
        if 'tickers' in losers_data:
            losers_list = []
            for ticker_data in losers_data['tickers']:
                if 'day' in ticker_data and 'prevDay' in ticker_data:
                    day_data = ticker_data['day']
                    prev_data = ticker_data['prevDay']
                    
                    losers_list.append({
                        'ticker': ticker_data['ticker'],
                        'change_percent': (day_data['c'] - prev_data['c']) / prev_data['c'] * 100,
                        'volume': day_data['v']
                    })
            losers_df = pd.DataFrame(losers_list)
        
        return gainers_df, losers_df
    
    def get_news_sentiment(self, ticker: str, from_date: str, to_date: str,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Get news articles and sentiment for a ticker
        
        Args:
            ticker: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Maximum number of articles to fetch
            
        Returns:
            DataFrame with news articles and sentiment
        """
        endpoint = f"/v2/reference/news"
        params = {
            'ticker': ticker,
            'published_utc.gte': from_date,
            'published_utc.lte': to_date,
            'limit': limit,
            'order': 'desc'
        }
        
        data = self._make_request(endpoint, params)
        
        if 'results' not in data:
            logger.warning(f"No news found for {ticker}")
            return pd.DataFrame()
            
        articles = []
        for article in data['results']:
            articles.append({
                'ticker': ticker,
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'published_utc': pd.to_datetime(article.get('published_utc')),
                'publisher': article.get('publisher', {}).get('name', ''),
                'keywords': article.get('keywords', []),
                'url': article.get('article_url', '')
            })
        
        return pd.DataFrame(articles)
    
    def get_market_indices_data(self, indices: List[str], from_date: str, to_date: str) -> Dict[str, pd.DataFrame]:
        """
        Get data for market-wide indices (SPY, QQQ, IWM, sector ETFs)
        
        Args:
            indices: List of index symbols
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping index symbols to DataFrames
        """
        index_data = {}
        
        for index_symbol in indices:
            try:
                df = self.get_aggregates(index_symbol, from_date, to_date)
                if not df.empty:
                    index_data[index_symbol] = df
                    logger.info(f"Fetched data for {index_symbol}: {len(df)} records")
                else:
                    logger.warning(f"No data found for {index_symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {index_symbol}: {e}")
                
        return index_data
    
    def get_comprehensive_market_data(self, tickers: List[str], indices: List[str], 
                                   from_date: str, to_date: str) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive market data for multiple tickers and indices
        
        Args:
            tickers: List of stock symbols
            indices: List of index symbols (SPY, QQQ, IWM, sector ETFs)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with market data for each ticker/index
        """
        all_data = {}
        
        # Fetch stock data
        for ticker in tickers:
            try:
                df = self.get_technical_indicators(ticker, from_date, to_date)
                if not df.empty:
                    all_data[ticker] = df
                    logger.info(f"Fetched data for {ticker}: {len(df)} records")
                else:
                    logger.warning(f"No data found for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        
        # Fetch index data
        index_data = self.get_market_indices_data(indices, from_date, to_date)
        all_data.update(index_data)
        
        return all_data
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str = None) -> bool:
        """
        Validate data quality and check for common issues
        
        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            logger.warning(f"No data returned for {symbol}")
            return False
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns for {symbol}: {missing_cols}")
            return False
        
        # Check for NaN values
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Data for {symbol} contains {nan_count} NaN values")
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                invalid_prices = (data[col] <= 0).sum()
                if invalid_prices > 0:
                    logger.warning(f"Data for {symbol} contains {invalid_prices} invalid prices in {col}")
        
        # Check for zero volume
        if 'volume' in data.columns:
            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > 0:
                logger.warning(f"Data for {symbol} contains {zero_volume} zero volume entries")
        
        logger.info(f"Data quality validation passed for {symbol}")
        return True


# Example usage and testing
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = PolygonDataFetcher()
    
    # Example: Fetch data for AAPL
    print("Fetching AAPL data...")
    aapl_data = fetcher.get_technical_indicators("AAPL", "2024-01-01", "2024-01-31")
    print(f"AAPL data shape: {aapl_data.shape}")
    print(aapl_data.head())
    
    # Example: Fetch market indices
    print("\nFetching market indices...")
    indices = ["SPY", "QQQ", "IWM", "XLK", "XLF"]
    index_data = fetcher.get_market_indices_data(indices, "2024-01-01", "2024-01-31")
    for symbol, df in index_data.items():
        print(f"{symbol}: {df.shape}")
    
    # Example: Fetch news sentiment
    print("\nFetching news sentiment...")
    news_data = fetcher.get_news_sentiment("AAPL", "2024-01-01", "2024-01-31", limit=10)
    print(f"News data shape: {news_data.shape}")
    if not news_data.empty:
        print(news_data[['title', 'published_utc', 'publisher']].head())
