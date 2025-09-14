"""
Data Storage System for Real-time LSTM Prediction System
Phase 1.2.3: Data Storage System with SQLite Database
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import logging
import os
import json

logger = logging.getLogger(__name__)

class DataStorage:
    """
    SQLite-based data storage system for market data and predictions
    """
    
    def __init__(self, db_path: str = "realtime_data.db"):
        """
        Initialize the data storage system
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
        
        logger.info(f"DataStorage initialized with database: {db_path}")
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            cursor = self.connection.cursor()
            
            # Create market_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            """)
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_timestamp DATETIME NOT NULL,
                    prediction_date DATETIME NOT NULL,
                    predicted_price REAL,
                    confidence_score REAL,
                    actual_price REAL,
                    prediction_error REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create trade_recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    recommendation_timestamp DATETIME NOT NULL,
                    entry_time DATETIME,
                    exit_time DATETIME,
                    entry_price REAL,
                    exit_price REAL,
                    expected_profit REAL,
                    expected_profit_percent REAL,
                    confidence_score REAL,
                    status TEXT DEFAULT 'active',
                    actual_entry_price REAL,
                    actual_exit_price REAL,
                    actual_profit REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create model_performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_version TEXT,
                    timestamp DATETIME NOT NULL,
                    rmse REAL,
                    mae REAL,
                    r2 REAL,
                    accuracy REAL,
                    total_predictions INTEGER,
                    correct_predictions INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create validation_results table (Phase 3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_timestamp DATETIME NOT NULL,
                    overall_status TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    requires_reprediction BOOLEAN NOT NULL,
                    confidence_adjustment REAL NOT NULL,
                    recommendations TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create validation_details table (Phase 3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_timestamp DATETIME NOT NULL,
                    validation_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL NOT NULL,
                    message TEXT,
                    details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create prediction_results table (Phase 3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_timestamp DATETIME NOT NULL,
                    task_id TEXT,
                    prediction_data TEXT NOT NULL,
                    validation_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create system_alerts table (Phase 3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create performance_metrics table (Phase 3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_timestamp DATETIME NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timestamp ON predictions(symbol, prediction_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_recommendations_symbol ON trade_recommendations(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_performance_symbol ON model_performance(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_validation_results_symbol_timestamp ON validation_results(symbol, prediction_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_validation_details_symbol_timestamp ON validation_details(symbol, prediction_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_results_symbol_timestamp ON prediction_results(symbol, prediction_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_alerts_type ON system_alerts(alert_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_symbol_timestamp ON performance_metrics(symbol, metric_timestamp)")
            
            self.connection.commit()
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def store_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store market data in the database
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Prepare data for insertion
            records = []
            for timestamp, row in data.iterrows():
                # Convert timestamp to string format that SQLite can handle
                if hasattr(timestamp, 'to_pydatetime'):
                    timestamp_str = timestamp.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
                elif hasattr(timestamp, 'strftime'):
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp_str = str(timestamp)
                
                records.append((
                    symbol,
                    timestamp_str,
                    float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                    float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                    float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                    float(row.get('close', 0)) if pd.notna(row.get('close')) else None,
                    int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None
                ))
            
            # Insert data (ignore duplicates)
            cursor.executemany("""
                INSERT OR IGNORE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            self.connection.commit()
            logger.info(f"Stored {len(records)} market data records for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing market data for {symbol}: {e}")
            return False
    
    def retrieve_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Retrieve historical market data
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with historical data
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (symbol, start_date, end_date))
            
            rows = cursor.fetchall()
            
            if not rows:
                # Only log warning if not in backtesting mode (backtest database)
                if not self.db_path.endswith('backtest_data.db'):
                    logger.warning(f"No historical data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append({
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} historical records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_data(self, symbol: str, hours_back: int = 24) -> pd.DataFrame:
        """
        Get latest market data for a symbol
        
        Args:
            symbol: Stock symbol
            hours_back: Hours to look back
            
        Returns:
            DataFrame with latest data
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Adjust for weekends - if we're on weekend, look back to last Friday
            if end_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                days_back = end_time.weekday() - 4  # Go back to Friday
                end_time = end_time - timedelta(days=days_back)
                end_time = end_time.replace(hour=16, minute=0, second=0, microsecond=0)  # 4 PM market close
                
                # Adjust start time accordingly
                start_time = end_time - timedelta(hours=hours_back)
                
                # Ensure start time doesn't go to weekend
                while start_time.weekday() >= 5:
                    start_time = start_time - timedelta(days=1)
                start_time = start_time.replace(hour=9, minute=30, second=0, microsecond=0)  # 9:30 AM market open
            
            logger.info(f"Retrieved {len(self.retrieve_historical_data(symbol, start_time, end_time))} historical records for {symbol}")
            return self.retrieve_historical_data(symbol, start_time, end_time)
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return pd.DataFrame()
    
    def store_prediction(self, symbol: str, prediction_timestamp: datetime, 
                        prediction_date: datetime, predicted_price: float, 
                        confidence_score: float, actual_price: float = None) -> bool:
        """
        Store a prediction in the database
        
        Args:
            symbol: Stock symbol
            prediction_timestamp: When the prediction was made
            prediction_date: What date the prediction is for
            predicted_price: Predicted price
            confidence_score: Confidence score (0-1)
            actual_price: Actual price (if available)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Calculate prediction error if actual price is available
            prediction_error = None
            if actual_price is not None:
                prediction_error = abs(predicted_price - actual_price) / actual_price
            
            # Convert datetime objects to strings for SQLite compatibility
            prediction_timestamp_str = prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(prediction_timestamp, 'strftime') else str(prediction_timestamp)
            prediction_date_str = prediction_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(prediction_date, 'strftime') else str(prediction_date)
            
            cursor.execute("""
                INSERT INTO predictions 
                (symbol, prediction_timestamp, prediction_date, predicted_price, 
                 confidence_score, actual_price, prediction_error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, prediction_timestamp_str, prediction_date_str, predicted_price, 
                  confidence_score, actual_price, prediction_error))
            
            self.connection.commit()
            logger.debug(f"Stored prediction for {symbol} at {prediction_date}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction for {symbol}: {e}")
            return False
    
    def store_trade_recommendation(self, symbol: str, entry_time: datetime, 
                                 exit_time: datetime, entry_price: float, 
                                 exit_price: float, confidence_score: float) -> int:
        """
        Store a trade recommendation
        
        Args:
            symbol: Stock symbol
            entry_time: Recommended entry time
            exit_time: Recommended exit time
            entry_price: Expected entry price
            exit_price: Expected exit price
            confidence_score: Confidence score
            
        Returns:
            Recommendation ID
        """
        try:
            cursor = self.connection.cursor()
            
            expected_profit = exit_price - entry_price
            expected_profit_percent = (expected_profit / entry_price) * 100
            
            cursor.execute("""
                INSERT INTO trade_recommendations 
                (symbol, recommendation_timestamp, entry_time, exit_time, 
                 entry_price, exit_price, expected_profit, expected_profit_percent, 
                 confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, datetime.now(), entry_time, exit_time, 
                  entry_price, exit_price, expected_profit, expected_profit_percent, 
                  confidence_score))
            
            recommendation_id = cursor.lastrowid
            self.connection.commit()
            
            logger.info(f"Stored trade recommendation {recommendation_id} for {symbol}")
            return recommendation_id
            
        except Exception as e:
            logger.error(f"Error storing trade recommendation for {symbol}: {e}")
            return -1
    
    def get_active_recommendations(self, symbol: str = None) -> pd.DataFrame:
        """
        Get active trade recommendations
        
        Args:
            symbol: Stock symbol (optional, if None returns all)
            
        Returns:
            DataFrame with active recommendations
        """
        try:
            cursor = self.connection.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM trade_recommendations
                    WHERE symbol = ? AND status = 'active'
                    ORDER BY recommendation_timestamp DESC
                """, (symbol,))
            else:
                cursor.execute("""
                    SELECT * FROM trade_recommendations
                    WHERE status = 'active'
                    ORDER BY recommendation_timestamp DESC
                """)
            
            rows = cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append(dict(row))
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error getting active recommendations: {e}")
            return pd.DataFrame()
    
    def update_recommendation_status(self, recommendation_id: int, status: str, 
                                   actual_entry_price: float = None, 
                                   actual_exit_price: float = None) -> bool:
        """
        Update recommendation status and actual prices
        
        Args:
            recommendation_id: Recommendation ID
            status: New status ('active', 'completed', 'cancelled')
            actual_entry_price: Actual entry price
            actual_exit_price: Actual exit price
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            actual_profit = None
            if actual_entry_price is not None and actual_exit_price is not None:
                actual_profit = actual_exit_price - actual_entry_price
            
            cursor.execute("""
                UPDATE trade_recommendations
                SET status = ?, actual_entry_price = ?, actual_exit_price = ?, actual_profit = ?
                WHERE id = ?
            """, (status, actual_entry_price, actual_exit_price, actual_profit, recommendation_id))
            
            self.connection.commit()
            logger.info(f"Updated recommendation {recommendation_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating recommendation {recommendation_id}: {e}")
            return False
    
    def cleanup_old_data(self, retention_days: int = 30) -> bool:
        """
        Clean up old data to manage database size
        
        Args:
            retention_days: Number of days to retain data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean up old market data
            cursor.execute("""
                DELETE FROM market_data 
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            market_deleted = cursor.rowcount
            
            # Clean up old predictions
            cursor.execute("""
                DELETE FROM predictions 
                WHERE prediction_timestamp < ?
            """, (cutoff_date,))
            
            predictions_deleted = cursor.rowcount
            
            self.connection.commit()
            
            logger.info(f"Cleaned up {market_deleted} market data records and {predictions_deleted} prediction records")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        try:
            cursor = self.connection.cursor()
            
            stats = {}
            
            # Count records in each table
            tables = ['market_data', 'predictions', 'trade_recommendations', 'model_performance']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            stats['database_size_bytes'] = cursor.fetchone()[0]
            
            # Get unique symbols
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data")
            stats['unique_symbols'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    # Phase 3: Enhanced data storage methods
    
    def store_validation_result(self, symbol: str, prediction_timestamp: datetime,
                              overall_status: str, overall_score: float,
                              requires_reprediction: bool, confidence_adjustment: float,
                              recommendations: List[str]) -> bool:
        """
        Store validation result (Phase 3)
        
        Args:
            symbol: Stock symbol
            prediction_timestamp: When the prediction was made
            overall_status: Overall validation status
            overall_score: Overall validation score
            requires_reprediction: Whether reprediction is required
            confidence_adjustment: Confidence adjustment factor
            recommendations: List of recommendations
            
        Returns:
            True if successful
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO validation_results 
                (symbol, prediction_timestamp, overall_status, overall_score,
                 requires_reprediction, confidence_adjustment, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, prediction_timestamp, overall_status, overall_score,
                  requires_reprediction, confidence_adjustment, json.dumps(recommendations)))
            
            self.connection.commit()
            logger.debug(f"Stored validation result for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing validation result for {symbol}: {e}")
            return False
    
    def store_validation_detail(self, symbol: str, prediction_timestamp: datetime,
                              validation_type: str, status: str, score: float,
                              message: str, details: Dict[str, Any]) -> bool:
        """
        Store validation detail (Phase 3)
        
        Args:
            symbol: Stock symbol
            prediction_timestamp: When the prediction was made
            validation_type: Type of validation
            status: Validation status
            score: Validation score
            message: Validation message
            details: Validation details
            
        Returns:
            True if successful
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO validation_details 
                (symbol, prediction_timestamp, validation_type, status, score, message, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, prediction_timestamp, validation_type, status, score, message, json.dumps(details)))
            
            self.connection.commit()
            logger.debug(f"Stored validation detail for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing validation detail for {symbol}: {e}")
            return False
    
    def store_prediction_result(self, symbol: str, prediction_result: Dict[str, Any],
                              validation_result: Any, task_id: str = None) -> bool:
        """
        Store complete prediction result (Phase 3)
        
        Args:
            symbol: Stock symbol
            prediction_result: Prediction result from PredictionEngine
            validation_result: Validation result
            task_id: Optional task ID
            
        Returns:
            True if successful
        """
        try:
            cursor = self.connection.cursor()
            
            # Serialize prediction and validation data
            prediction_data = json.dumps(prediction_result, default=str)
            validation_data = json.dumps(validation_result.__dict__, default=str) if validation_result else None
            
            cursor.execute("""
                INSERT INTO prediction_results 
                (symbol, prediction_timestamp, task_id, prediction_data, validation_data)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, prediction_result['prediction_timestamp'], task_id, prediction_data, validation_data))
            
            self.connection.commit()
            logger.info(f"Stored prediction result for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction result for {symbol}: {e}")
            return False
    
    def store_system_alert(self, alert_type: str, severity: str, message: str,
                          details: Dict[str, Any] = None) -> int:
        """
        Store system alert (Phase 3)
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            details: Optional alert details
            
        Returns:
            Alert ID
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO system_alerts 
                (alert_type, severity, message, details)
                VALUES (?, ?, ?, ?)
            """, (alert_type, severity, message, json.dumps(details) if details else None))
            
            alert_id = cursor.lastrowid
            self.connection.commit()
            
            logger.info(f"Stored system alert {alert_id}: {message}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error storing system alert: {e}")
            return -1
    
    def store_performance_metric(self, symbol: str, metric_name: str, metric_value: float,
                               metric_timestamp: datetime, metadata: Dict[str, Any] = None) -> bool:
        """
        Store performance metric (Phase 3)
        
        Args:
            symbol: Stock symbol (optional)
            metric_name: Name of the metric
            metric_value: Metric value
            metric_timestamp: When the metric was recorded
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        try:
            # Ensure we have a valid connection
            if not self.connection:
                self._initialize_database()
            
            cursor = self.connection.cursor()
            
            # Convert datetime to string for SQLite compatibility
            timestamp_str = metric_timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(metric_timestamp, 'strftime') else str(metric_timestamp)
            
            # Simple insert without explicit transaction management
            cursor.execute("""
                INSERT INTO performance_metrics 
                (symbol, metric_name, metric_value, metric_timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, metric_name, metric_value, timestamp_str, json.dumps(metadata) if metadata else '{}'))
            
            self.connection.commit()
            symbol_display = symbol if symbol else "system"
            logger.debug(f"Stored performance metric {metric_name} for {symbol_display}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing performance metric: {e}")
            return False
    
    def get_validation_history(self, symbol: str, days_back: int = 7) -> pd.DataFrame:
        """
        Get validation history for a symbol (Phase 3)
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            DataFrame with validation history
        """
        try:
            cursor = self.connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            cursor.execute("""
                SELECT * FROM validation_results
                WHERE symbol = ? AND prediction_timestamp >= ?
                ORDER BY prediction_timestamp DESC
            """, (symbol, cutoff_date))
            
            rows = cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append(dict(row))
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error getting validation history for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(self, symbol: str = None, metric_name: str = None,
                               days_back: int = 30) -> pd.DataFrame:
        """
        Get performance metrics (Phase 3)
        
        Args:
            symbol: Stock symbol (optional)
            metric_name: Metric name (optional)
            days_back: Number of days to look back
            
        Returns:
            DataFrame with performance metrics
        """
        try:
            cursor = self.connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = """
                SELECT * FROM performance_metrics
                WHERE metric_timestamp >= ?
            """
            params = [cutoff_date]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            
            query += " ORDER BY metric_timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append(dict(row))
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return pd.DataFrame()
    
    def get_system_alerts(self, alert_type: str = None, resolved: bool = None,
                         days_back: int = 7) -> pd.DataFrame:
        """
        Get system alerts (Phase 3)
        
        Args:
            alert_type: Alert type (optional)
            resolved: Whether to get resolved/unresolved alerts (optional)
            days_back: Number of days to look back
            
        Returns:
            DataFrame with system alerts
        """
        try:
            cursor = self.connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = """
                SELECT * FROM system_alerts
                WHERE created_at >= ?
            """
            params = [cutoff_date]
            
            if alert_type:
                query += " AND alert_type = ?"
                params.append(alert_type)
            
            if resolved is not None:
                query += " AND resolved = ?"
                params.append(resolved)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append(dict(row))
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            logger.error(f"Error getting system alerts: {e}")
            return pd.DataFrame()
    
    def resolve_alert(self, alert_id: int) -> bool:
        """
        Resolve a system alert (Phase 3)
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if successful
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                UPDATE system_alerts
                SET resolved = TRUE, resolved_at = ?
                WHERE id = ?
            """, (datetime.now(), alert_id))
            
            self.connection.commit()
            logger.info(f"Resolved alert {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def get_enhanced_database_stats(self) -> Dict[str, Any]:
        """
        Get enhanced database statistics including Phase 3 tables
        
        Returns:
            Dictionary with comprehensive database statistics
        """
        try:
            cursor = self.connection.cursor()
            
            stats = {}
            
            # Count records in each table
            tables = ['market_data', 'predictions', 'trade_recommendations', 'model_performance',
                     'validation_results', 'validation_details', 'prediction_results',
                     'system_alerts', 'performance_metrics']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            stats['database_size_bytes'] = cursor.fetchone()[0]
            
            # Get unique symbols
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data")
            stats['unique_symbols'] = cursor.fetchone()[0]
            
            # Get recent activity (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_timestamp >= ?", (cutoff_time,))
            stats['predictions_last_24h'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM validation_results WHERE prediction_timestamp >= ?", (cutoff_time,))
            stats['validations_last_24h'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM system_alerts WHERE created_at >= ?", (cutoff_time,))
            stats['alerts_last_24h'] = cursor.fetchone()[0]
            
            # Get unresolved alerts
            cursor.execute("SELECT COUNT(*) FROM system_alerts WHERE resolved = FALSE")
            stats['unresolved_alerts'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting enhanced database stats: {e}")
            return {}
    
    def store_error_record(self, error_id: str, timestamp: datetime, severity: str, 
                          category: str, error_type: str, message: str, 
                          stack_trace: str = None, metadata: dict = None) -> bool:
        """
        Store error record in database
        
        Args:
            error_id: Unique error ID
            timestamp: Error timestamp
            severity: Error severity level
            category: Error category
            error_type: Type of error
            message: Error message
            stack_trace: Stack trace (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Create errors table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_records (
                    error_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    stack_trace TEXT,
                    metadata TEXT
                )
            """)
            
            # Insert error record
            cursor.execute("""
                INSERT OR REPLACE INTO error_records 
                (error_id, timestamp, severity, category, error_type, message, stack_trace, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (error_id, timestamp.isoformat(), severity, category, error_type, 
                  message, stack_trace, json.dumps(metadata) if metadata else None))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing error record: {e}")
            return False
    
    def store_performance_profile(self, profile_id: str, timestamp: datetime, 
                                function_name: str, execution_time: float,
                                memory_usage: float = None, metadata: dict = None) -> bool:
        """
        Store performance profile in database
        
        Args:
            profile_id: Unique profile ID
            timestamp: Profile timestamp
            function_name: Name of profiled function
            execution_time: Execution time in seconds
            memory_usage: Memory usage in MB (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Create performance profiles table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_profiles (
                    profile_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    memory_usage REAL,
                    metadata TEXT
                )
            """)
            
            # Insert performance profile
            cursor.execute("""
                INSERT OR REPLACE INTO performance_profiles 
                (profile_id, timestamp, function_name, execution_time, memory_usage, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (profile_id, timestamp.isoformat(), function_name, execution_time, 
                  memory_usage, json.dumps(metadata) if metadata else None))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing performance profile: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the storage system
    storage = DataStorage("test_data.db")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0],
        'high': [101.0, 102.0, 103.0],
        'low': [99.0, 100.0, 101.0],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=3, freq='15min'))
    
    # Test storing data
    success = storage.store_market_data('AAPL', sample_data)
    print(f"Data storage successful: {success}")
    
    # Test retrieving data
    retrieved_data = storage.get_latest_data('AAPL', hours_back=1)
    print(f"Retrieved data shape: {retrieved_data.shape}")
    
    # Test storing prediction
    storage.store_prediction('AAPL', datetime.now(), datetime.now() + timedelta(days=1), 
                           105.0, 0.85, 104.5)
    
    # Test storing trade recommendation
    rec_id = storage.store_trade_recommendation('AAPL', datetime.now(), 
                                              datetime.now() + timedelta(days=3),
                                              100.0, 105.0, 0.8)
    print(f"Trade recommendation ID: {rec_id}")
    
    # Get database stats
    stats = storage.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Cleanup
    storage.close()
    os.remove("test_data.db")
