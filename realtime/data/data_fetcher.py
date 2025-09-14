"""
Polygon.io Data Fetcher for Real-time LSTM Prediction System
Phase 1.2.1: Polygon.io Integration with Rate Limiting & Error Handling
"""

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

logger = logging.getLogger(__name__)

class PolygonDataFetcher:
    """
    Enhanced Polygon.io data fetcher with rate limiting, error handling, and caching
    """
    
    def __init__(self, api_key: str = None, rate_limit: int = 100):
        """
        Initialize the Polygon.io data fetcher
        
        Args:
            api_key: Polygon.io API key. If None, will try to get from environment variable
            rate_limit: Maximum API calls per minute (default: 100)
        """
        if api_key is None:
            api_key = os.getenv('POLYGON_API_KEY')
            if api_key is None:
                raise ValueError("Polygon.io API key is required. Set POLYGON_API_KEY environment variable or pass api_key parameter.")
        
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.base_url = "https://api.polygon.io"
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"PolygonDataFetcher initialized with rate limit: {rate_limit} calls/minute")
    
    def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check if we've hit the rate limit
        if self.request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = time.time()
        
        # Ensure minimum time between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < 0.6:  # 600ms between requests
            time.sleep(0.6 - time_since_last)
        
        self.request_count += 1
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a rate-limited request to Polygon.io API
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response as dictionary
        """
        self._rate_limit_check()
        
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.debug(f"Making request to: {url}")
            logger.debug(f"Request params: {params}")
            
            response = self.session.get(url, params=params, timeout=30)
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Response data: {data}")
            
            if data.get('status') == 'OK':
                return data
            else:
                error_msg = data.get('message', 'Unknown API error')
                logger.error(f"Polygon API error: {error_msg}")
                logger.error(f"Full API response: {data}")
                raise Exception(f"Polygon API error: {error_msg}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            logger.error(f"Request URL: {url}")
            logger.error(f"Request params: {params}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(f"Request URL: {url}")
            logger.error(f"Request params: {params}")
            raise
    
    def fetch_15min_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Fetch 15-minute bar data from Polygon.io
        
        Args:
            symbol: Stock ticker symbol
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching 15-minute data for {symbol} from {start_time} to {end_time}")
            
            # Format dates for API
            start_str = start_time.strftime('%Y-%m-%d')
            end_str = end_time.strftime('%Y-%m-%d')
            
            endpoint = f"/v2/aggs/ticker/{symbol}/range/15/minute/{start_str}/{end_str}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000
            }
            
            data = self._make_request(endpoint, params)
            
            if not data.get('results'):
                logger.warning(f"No 15-minute data found for {symbol} from {start_str} to {end_str}")
                
                # Try fallback: get data from previous trading day
                logger.info(f"Trying fallback: previous trading day for {symbol}")
                fallback_end = start_time - timedelta(days=1)
                fallback_start = fallback_end - timedelta(days=1)
                
                # Ensure fallback doesn't go to weekend
                while fallback_end.weekday() >= 5:
                    fallback_end = fallback_end - timedelta(days=1)
                while fallback_start.weekday() >= 5:
                    fallback_start = fallback_start - timedelta(days=1)
                
                fallback_start_str = fallback_start.strftime('%Y-%m-%d')
                fallback_end_str = fallback_end.strftime('%Y-%m-%d')
                
                endpoint = f"/v2/aggs/ticker/{symbol}/range/15/minute/{fallback_start_str}/{fallback_end_str}"
                data = self._make_request(endpoint, params)
                
                if not data.get('results'):
                    logger.warning(f"No fallback data found for {symbol}")
                    return pd.DataFrame()
            
            # Convert to DataFrame
            bars = []
            for bar in data['results']:
                bars.append({
                    'timestamp': pd.to_datetime(bar['t'], unit='ms'),
                    'open': bar['o'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar['v']
                })
            
            df = pd.DataFrame(bars)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Retrieved {len(df)} 15-minute bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching 15-minute data for {symbol}: {e}")
            raise
    
    def fetch_latest_data(self, symbol: str, lookback_hours: int = 24) -> pd.DataFrame:
        """
        Fetch latest data for a symbol
        
        Args:
            symbol: Stock ticker symbol
            lookback_hours: Hours to look back (default: 24)
            
        Returns:
            DataFrame with latest OHLCV data
        """
        try:
            # Get current time and adjust for weekends
            current_time = datetime.now()
            
            # If it's weekend, go back to last Friday
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                days_back = current_time.weekday() - 4  # Go back to Friday
                current_time = current_time - timedelta(days=days_back)
                current_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)  # 4 PM market close
            
            end_time = current_time
            start_time = end_time - timedelta(hours=lookback_hours)
            
            # Ensure we don't go back to weekend
            if start_time.weekday() >= 5:
                # Go back to previous Friday
                days_back = start_time.weekday() - 4
                start_time = start_time - timedelta(days=days_back)
                start_time = start_time.replace(hour=9, minute=30, second=0, microsecond=0)  # 9:30 AM market open
            
            logger.info(f"Adjusted time range for {symbol}: {start_time} to {end_time}")
            return self.fetch_15min_data(symbol, start_time, end_time)
            
        except Exception as e:
            logger.error(f"Error fetching latest data for {symbol}: {e}")
            raise
    
    def fetch_2_years_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch 2 years of historical data for comprehensive model training/prediction
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with 2 years of OHLCV data
        """
        try:
            # Get current time and adjust for weekends
            current_time = datetime.now()
            
            # If it's weekend, go back to last Friday
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                days_back = current_time.weekday() - 4  # Go back to Friday
                current_time = current_time - timedelta(days=days_back)
                current_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)  # 4 PM market close
            
            end_time = current_time
            start_time = end_time - timedelta(days=730)  # 2 years = ~730 days
            
            # Ensure we don't go back to weekend
            if start_time.weekday() >= 5:
                # Go back to previous Friday
                days_back = start_time.weekday() - 4
                start_time = start_time - timedelta(days=days_back)
                start_time = start_time.replace(hour=9, minute=30, second=0, microsecond=0)  # 9:30 AM market open
            
            logger.info(f"Fetching 2 years of historical data for {symbol}: {start_time} to {end_time}")
            
            # Fetch data in chunks to avoid API limits
            all_data = []
            chunk_start = start_time
            chunk_days = 90  # 3 months per chunk
            
            while chunk_start < end_time:
                chunk_end = min(chunk_start + timedelta(days=chunk_days), end_time)
                
                logger.info(f"Fetching chunk: {chunk_start} to {chunk_end}")
                chunk_data = self.fetch_15min_data(symbol, chunk_start, chunk_end)
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                
                chunk_start = chunk_end + timedelta(minutes=15)  # Small overlap to avoid gaps
                
                # Rate limiting - wait between chunks
                time.sleep(1)
            
            if not all_data:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Combine all chunks
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index().drop_duplicates()
            
            logger.info(f"Retrieved {len(combined_data)} total historical records for {symbol} over 2 years")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error fetching 2 years historical data for {symbol}: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current stock price
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Current price or None if not available
        """
        try:
            # Try to get latest trade
            endpoint = f"/v2/last/trade/{symbol}"
            data = self._make_request(endpoint)
            
            if data.get('results'):
                return float(data['results']['p'])
            
            # Fallback: get latest close from 15-minute data
            latest_data = self.fetch_latest_data(symbol, lookback_hours=1)
            if not latest_data.empty:
                return float(latest_data['close'].iloc[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status
        
        Returns:
            Dictionary with market status information
        """
        try:
            endpoint = "/v1/marketstatus/now"
            data = self._make_request(endpoint)
            
            return {
                'market': data.get('market', 'unknown'),
                'serverTime': data.get('serverTime'),
                'exchanges': data.get('exchanges', {}),
                'currencies': data.get('currencies', {})
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {}
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and is tradeable
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            endpoint = f"/v3/reference/tickers/{symbol}"
            data = self._make_request(endpoint)
            
            if data.get('results'):
                ticker_info = data['results']
                return ticker_info.get('active', False) and ticker_info.get('type') == 'CS'
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with symbol information
        """
        try:
            endpoint = f"/v3/reference/tickers/{symbol}"
            data = self._make_request(endpoint)
            
            if data.get('results'):
                return data['results']
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the fetcher
    fetcher = PolygonDataFetcher()
    
    # Test symbol validation
    test_symbols = ['AAPL', 'GOOGL', 'INVALID_SYMBOL']
    for symbol in test_symbols:
        is_valid = fetcher.validate_symbol(symbol)
        print(f"{symbol}: {'Valid' if is_valid else 'Invalid'}")
    
    # Test data fetching
    try:
        data = fetcher.fetch_latest_data('AAPL', lookback_hours=2)
        print(f"Latest AAPL data shape: {data.shape}")
        print(f"Latest AAPL data:\n{data.tail()}")
    except Exception as e:
        print(f"Error fetching data: {e}")
