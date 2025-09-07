# ==============================================================================
# Polygon.io Data Fetcher
# Replaces Yahoo Finance with Polygon.io for more reliable and comprehensive data

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon import RESTClient
import os
from typing import Optional, Dict, Any


class PolygonDataFetcher:
    """
    A class to fetch stock data from Polygon.io API
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Polygon.io data fetcher
        
        Args:
            api_key: Polygon.io API key. If None, will try to get from environment variable POLYGON_API_KEY
        """
        if api_key is None:
            api_key = os.getenv('POLYGON_API_KEY')
            if api_key is None:
                raise ValueError("Polygon.io API key is required. Set POLYGON_API_KEY environment variable or pass api_key parameter.")
        
        self.api_key = api_key
        self.client = RESTClient(api_key)
        
    def get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """
        Get historical stock data for a given ticker
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date for data
            end_date: End date for data (defaults to today)
            
        Returns:
            pandas.DataFrame: Stock data with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        if end_date is None:
            end_date = datetime.now()
            
        # Format dates for Polygon.io API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        try:
            # Get aggregates (bars) data from Polygon.io
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_str,
                to=end_str,
                adjusted=True,
                sort="asc",
                limit=50000
            )
            
            if not aggs:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'Date': pd.to_datetime(agg.timestamp, unit='ms'),
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            raise
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get stock information from Polygon.io
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            dict: Stock information
        """
        try:
            # Get ticker details
            ticker_details = self.client.get_ticker_details(ticker)
            
            info = {
                'name': getattr(ticker_details, 'name', ''),
                'market': getattr(ticker_details, 'market', ''),
                'locale': getattr(ticker_details, 'locale', ''),
                'primary_exchange': getattr(ticker_details, 'primary_exchange', ''),
                'type': getattr(ticker_details, 'type', ''),
                'active': getattr(ticker_details, 'active', True),
                'currency_name': getattr(ticker_details, 'currency_name', 'USD'),
                'cik': getattr(ticker_details, 'cik', ''),
                'composite_figi': getattr(ticker_details, 'composite_figi', ''),
                'share_class_figi': getattr(ticker_details, 'share_class_figi', ''),
                'market_cap': getattr(ticker_details, 'market_cap', 0),
                'phone_number': getattr(ticker_details, 'phone_number', ''),
                'address': getattr(ticker_details, 'address', {}),
                'description': getattr(ticker_details, 'description', ''),
                'sic_code': getattr(ticker_details, 'sic_code', ''),
                'sic_description': getattr(ticker_details, 'sic_description', ''),
                'ticker_root': getattr(ticker_details, 'ticker_root', ''),
                'homepage_url': getattr(ticker_details, 'homepage_url', ''),
                'total_employees': getattr(ticker_details, 'total_employees', 0),
                'list_date': getattr(ticker_details, 'list_date', ''),
                'branding': getattr(ticker_details, 'branding', {}),
                'share_class_shares_outstanding': getattr(ticker_details, 'share_class_shares_outstanding', 0),
                'weighted_shares_outstanding': getattr(ticker_details, 'weighted_shares_outstanding', 0),
                'round_lot': getattr(ticker_details, 'round_lot', 100)
            }
            
            return info
            
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return {}
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get the current stock price
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            float: Current stock price or None if not available
        """
        try:
            # Get the latest trade
            trades = self.client.list_trades(ticker, limit=1)
            
            if trades:
                return float(trades[0].price)
            else:
                # Fallback: get latest close from daily data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=5)  # Get last few days
                
                df = self.get_stock_data(ticker, start_date, end_date)
                if not df.empty:
                    return float(df['Close'].iloc[-1])
                    
            return None
            
        except Exception as e:
            print(f"Error getting current price for {ticker}: {e}")
            return None
    
    def get_ticker_list(self, market: str = "stocks", active: bool = True, limit: int = 1000) -> pd.DataFrame:
        """
        Get list of available tickers
        
        Args:
            market: Market type (stocks, crypto, fx, etc.)
            active: Whether to get only active tickers
            limit: Maximum number of tickers to return
            
        Returns:
            pandas.DataFrame: List of tickers
        """
        try:
            tickers = self.client.list_tickers(market=market, active=active, limit=limit)
            
            data = []
            for ticker in tickers:
                data.append({
                    'ticker': ticker.ticker,
                    'name': getattr(ticker, 'name', ''),
                    'market': getattr(ticker, 'market', ''),
                    'locale': getattr(ticker, 'locale', ''),
                    'primary_exchange': getattr(ticker, 'primary_exchange', ''),
                    'type': getattr(ticker, 'type', ''),
                    'active': getattr(ticker, 'active', True),
                    'currency_name': getattr(ticker, 'currency_name', 'USD'),
                    'cik': getattr(ticker, 'cik', ''),
                    'composite_figi': getattr(ticker, 'composite_figi', ''),
                    'share_class_figi': getattr(ticker, 'share_class_figi', ''),
                    'last_updated_utc': getattr(ticker, 'last_updated_utc', '')
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error fetching ticker list: {e}")
            return pd.DataFrame()


# Compatibility wrapper to match yfinance interface
class PolygonTicker:
    """
    A compatibility wrapper to match yfinance.Ticker interface
    """
    
    def __init__(self, ticker: str, api_key: str = None):
        self.ticker = ticker
        self.fetcher = PolygonDataFetcher(api_key)
        self._info = None
        self._history_data = None
    
    @property
    def info(self) -> Dict[str, Any]:
        """Get stock information (compatible with yfinance)"""
        if self._info is None:
            self._info = self.fetcher.get_stock_info(self.ticker)
        return self._info
    
    def history(self, period: str = "1y", start: str = None, end: str = None) -> pd.DataFrame:
        """
        Get historical data (compatible with yfinance)
        
        Args:
            period: Period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            
        Returns:
            pandas.DataFrame: Historical data
        """
        # Parse period or use start/end dates
        if start and end:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
        else:
            # Map period to date range
            end_date = datetime.now()
            if period == "1d":
                start_date = end_date - timedelta(days=1)
            elif period == "5d":
                start_date = end_date - timedelta(days=5)
            elif period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                start_date = end_date - timedelta(days=730)
            elif period == "5y":
                start_date = end_date - timedelta(days=1825)
            elif period == "10y":
                start_date = end_date - timedelta(days=3650)
            elif period == "ytd":
                start_date = datetime(end_date.year, 1, 1)
            elif period == "max":
                start_date = datetime(2000, 1, 1)  # Polygon.io data goes back to around 2000
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        return self.fetcher.get_stock_data(self.ticker, start_date, end_date)


# Convenience function to match yfinance.download interface
def download(tickers: list, start: datetime = None, end: datetime = None, 
             api_key: str = None, **kwargs) -> pd.DataFrame:
    """
    Download stock data for multiple tickers (compatible with yfinance.download)
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        api_key: Polygon.io API key
        **kwargs: Additional arguments (ignored for compatibility)
        
    Returns:
        pandas.DataFrame: Stock data
    """
    if start is None:
        start = datetime.now() - timedelta(days=365)  # Default to 1 year
    if end is None:
        end = datetime.now()
    
    fetcher = PolygonDataFetcher(api_key)
    
    if len(tickers) == 1:
        # Single ticker
        ticker = tickers[0]
        data = fetcher.get_stock_data(ticker, start, end)
        # Add ticker column for consistency with yfinance
        data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
        return data
    else:
        # Multiple tickers
        all_data = {}
        for ticker in tickers:
            try:
                data = fetcher.get_stock_data(ticker, start, end)
                all_data[ticker] = data
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue
        
        if all_data:
            # Combine all data
            combined = pd.concat(all_data.values(), keys=all_data.keys(), axis=1)
            return combined
        else:
            return pd.DataFrame()
