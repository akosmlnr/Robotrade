"""
Historical Data Fetcher for Backtesting
Fetches historical data for random week selection and simulation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import random
import os
import sys

# Add parent directory to path to import realtime modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_fetcher import PolygonDataFetcher

logger = logging.getLogger(__name__)

class HistoricalDataFetcher:
    """
    Enhanced data fetcher for backtesting that can select random historical periods
    and provide incremental data updates for simulation
    """
    
    def __init__(self, api_key: str = None, rate_limit: int = 100):
        """
        Initialize the historical data fetcher
        
        Args:
            api_key: Polygon.io API key. If None, will try to get from environment variable
            rate_limit: Maximum API calls per minute (default: 100)
        """
        self.data_fetcher = PolygonDataFetcher(api_key=api_key, rate_limit=rate_limit)
        self.cache = {}  # Cache for historical data
        
        logger.info("HistoricalDataFetcher initialized")
    
    def get_random_historical_week(self, symbol: str, 
                                 min_date: datetime = None, 
                                 max_date: datetime = None) -> Tuple[datetime, datetime]:
        """
        Select a random historical week for backtesting
        
        Args:
            symbol: Stock symbol
            min_date: Minimum date for selection (default: 2 years ago)
            max_date: Maximum date for selection (default: 1 year ago)
            
        Returns:
            Tuple of (start_date, end_date) for the selected week
        """
        try:
            # Default date ranges if not provided
            if max_date is None:
                max_date = datetime.now() - timedelta(days=365)  # 1 year ago
            if min_date is None:
                min_date = max_date - timedelta(days=365)  # 2 years ago
            
            # Ensure we have valid business days
            min_date = self._adjust_to_business_day(min_date, forward=True)
            max_date = self._adjust_to_business_day(max_date, forward=False)
            
            # Calculate total available weeks
            total_days = (max_date - min_date).days
            total_weeks = total_days // 7
            
            if total_weeks < 1:
                raise ValueError("Insufficient date range for week selection")
            
            # Select a random week
            random_week = random.randint(0, total_weeks - 1)
            start_date = min_date + timedelta(weeks=random_week)
            end_date = start_date + timedelta(days=7)
            
            # Adjust to business days
            start_date = self._adjust_to_business_day(start_date, forward=True)
            end_date = self._adjust_to_business_day(end_date, forward=False)
            
            logger.info(f"Selected random week for {symbol}: {start_date.date()} to {end_date.date()}")
            return start_date, end_date
            
        except Exception as e:
            logger.error(f"Error selecting random historical week for {symbol}: {e}")
            # Fallback to a default week
            fallback_end = datetime.now() - timedelta(days=365)
            fallback_start = fallback_end - timedelta(days=7)
            return fallback_start, fallback_end
    
    def fetch_historical_week_data(self, symbol: str, start_date: datetime, 
                                 end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data for a specific week
        
        Args:
            symbol: Stock symbol
            start_date: Start date of the week
            end_date: End date of the week
            
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            # Create cache key
            cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            # Check cache first
            if cache_key in self.cache:
                logger.info(f"Using cached data for {symbol} week {start_date.date()} to {end_date.date()}")
                return self.cache[cache_key].copy()
            
            # Fetch data from API
            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            # Extend the range to ensure we get enough data (add buffer for sequence length and historical context)
            buffer_start = start_date - timedelta(days=60)  # 60 days buffer for historical context (30 trading days + weekends + holidays)
            buffer_end = end_date + timedelta(days=45)      # 45 days buffer for prediction period (30 trading days + weekends)
            
            historical_data = self.data_fetcher.fetch_15min_data(
                symbol, buffer_start, buffer_end
            )
            
            if historical_data.empty:
                logger.warning(f"No historical data found for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Filter to include the week plus enough days after for validation
            # This ensures we have actual data for the predicted time periods (30 trading days)
            validation_end = end_date + timedelta(days=45)  # Include 6 weeks after for validation
            week_data = historical_data[
                (historical_data.index >= buffer_start) & 
                (historical_data.index <= validation_end)
            ].copy()
            
            # Cache the result
            self.cache[cache_key] = week_data.copy()
            
            logger.info(f"Retrieved {len(week_data)} data points for {symbol} week")
            return week_data
            
        except Exception as e:
            logger.error(f"Error fetching historical week data for {symbol}: {e}")
            return pd.DataFrame()
    
    def simulate_incremental_data_updates(self, symbol: str, historical_data: pd.DataFrame,
                                        start_time: datetime, 
                                        update_interval_minutes: int = 15) -> List[Dict[str, Any]]:
        """
        Simulate incremental data updates by providing data in chunks
        
        Args:
            symbol: Stock symbol
            historical_data: Full historical data for the week
            start_time: Starting time for the simulation
            update_interval_minutes: Minutes between updates (default: 15)
            
        Returns:
            List of data updates, each containing timestamp and available data
        """
        try:
            if historical_data.empty:
                logger.warning(f"No historical data provided for {symbol}")
                return []
            
            updates = []
            current_time = start_time
            
            # Sort data by timestamp
            sorted_data = historical_data.sort_index()
            
            # Create incremental updates every 15 minutes
            while current_time <= sorted_data.index[-1]:
                # Get all data up to current time
                available_data = sorted_data[sorted_data.index <= current_time]
                
                if not available_data.empty:
                    update = {
                        'timestamp': current_time,
                        'symbol': symbol,
                        'available_data': available_data.copy(),
                        'latest_price': available_data['close'].iloc[-1],
                        'data_points': len(available_data),
                        'is_new_data': len(available_data) > (len(updates) * update_interval_minutes // 15) if updates else True
                    }
                    updates.append(update)
                    
                    # Prevent excessive memory usage by limiting updates
                    if len(updates) > 10000:  # Reasonable limit for a week of 15-min data
                        logger.warning(f"Too many updates generated ({len(updates)}), stopping simulation")
                        break
                    
                    logger.debug(f"Created update for {symbol} at {current_time}: {len(available_data)} data points, "
                               f"latest price: ${update['latest_price']:.2f}")
                
                # Move to next 15-minute interval
                current_time += timedelta(minutes=update_interval_minutes)
            
            logger.info(f"Created {len(updates)} incremental updates for {symbol}")
            return updates
            
        except Exception as e:
            logger.error(f"Error simulating incremental data updates for {symbol}: {e}")
            return []
    
    def get_prediction_sequence_data(self, available_data: pd.DataFrame, 
                                   sequence_length: int = 25) -> Optional[pd.DataFrame]:
        """
        Get data suitable for creating prediction sequences
        
        Args:
            available_data: Available historical data
            sequence_length: Required sequence length for LSTM
            
        Returns:
            DataFrame with enough data for sequence creation, or None if insufficient
        """
        try:
            if len(available_data) < sequence_length:
                logger.warning(f"Insufficient data for sequence: need {sequence_length}, got {len(available_data)}")
                return None
            
            # Return the last sequence_length data points
            return available_data.tail(sequence_length).copy()
            
        except Exception as e:
            logger.error(f"Error getting prediction sequence data: {e}")
            return None
    
    def calculate_week_statistics(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics for the historical week
        
        Args:
            historical_data: Historical data for the week
            
        Returns:
            Dictionary with week statistics
        """
        try:
            if historical_data.empty:
                return {}
            
            # Basic statistics
            stats = {
                'total_data_points': len(historical_data),
                'start_time': historical_data.index[0],
                'end_time': historical_data.index[-1],
                'duration_hours': (historical_data.index[-1] - historical_data.index[0]).total_seconds() / 3600,
                'price_range': {
                    'min': historical_data['close'].min(),
                    'max': historical_data['close'].max(),
                    'range': historical_data['close'].max() - historical_data['close'].min(),
                    'range_percent': ((historical_data['close'].max() - historical_data['close'].min()) / max(historical_data['close'].min(), 0.01)) * 100
                },
                'volatility': {
                    'daily_returns_std': historical_data['close'].pct_change().std(),
                    'price_std': historical_data['close'].std()
                },
                'volume_stats': {
                    'avg_volume': historical_data['volume'].mean(),
                    'max_volume': historical_data['volume'].max(),
                    'min_volume': historical_data['volume'].min()
                }
            }
            
            # Calculate price change over the week
            if len(historical_data) > 1:
                first_price = historical_data['close'].iloc[0]
                last_price = historical_data['close'].iloc[-1]
                stats['weekly_change'] = {
                    'absolute': last_price - first_price,
                    'percent': ((last_price - first_price) / max(first_price, 0.01)) * 100
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating week statistics: {e}")
            return {}
    
    def _adjust_to_business_day(self, date: datetime, forward: bool = True) -> datetime:
        """
        Adjust date to nearest business day
        
        Args:
            date: Date to adjust
            forward: If True, move forward to next business day; if False, move backward
            
        Returns:
            Adjusted datetime
        """
        try:
            # Monday = 0, Sunday = 6
            weekday = date.weekday()
            
            if weekday < 5:  # Monday to Friday
                return date
            
            if forward:
                # Move to next Monday
                days_ahead = 7 - weekday
                return date + timedelta(days=days_ahead)
            else:
                # Move to previous Friday
                days_back = weekday - 4
                return date - timedelta(days=days_back)
                
        except Exception as e:
            logger.error(f"Error adjusting to business day: {e}")
            return date
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        logger.info("Historical data cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache"""
        return {
            'cache_size': len(self.cache),
            'cached_symbols': list(set(key.split('_')[0] for key in self.cache.keys())),
            'total_data_points': sum(len(df) for df in self.cache.values())
        }


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the historical data fetcher
    fetcher = HistoricalDataFetcher()
    
    # Test random week selection
    symbol = "AAPL"
    start_date, end_date = fetcher.get_random_historical_week(symbol)
    print(f"Selected week for {symbol}: {start_date.date()} to {end_date.date()}")
    
    # Test data fetching
    try:
        historical_data = fetcher.fetch_historical_week_data(symbol, start_date, end_date)
        print(f"Retrieved {len(historical_data)} data points")
        
        # Test incremental updates
        updates = fetcher.simulate_incremental_data_updates(symbol, historical_data, start_date)
        print(f"Created {len(updates)} incremental updates")
        
        # Test statistics
        stats = fetcher.calculate_week_statistics(historical_data)
        print(f"Week statistics: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
