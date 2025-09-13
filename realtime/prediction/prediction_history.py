"""
Prediction History Tracking for Real-time LSTM Prediction System
Phase 3.1: Comprehensive tracking and analysis of prediction history
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class PredictionStatus(Enum):
    """Status of predictions"""
    ACTIVE = "active"
    EXPIRED = "expired"
    VALIDATED = "validated"
    INVALIDATED = "invalidated"

@dataclass
class PredictionRecord:
    """Record of a prediction"""
    prediction_id: str
    symbol: str
    prediction_timestamp: datetime
    prediction_horizon: str
    predicted_prices: List[float]
    prediction_timestamps: List[datetime]
    confidence_score: float
    model_version: str
    validation_status: Optional[str] = None
    validation_score: Optional[float] = None
    actual_prices: Optional[List[float]] = None
    actual_timestamps: Optional[List[datetime]] = None
    accuracy_metrics: Optional[Dict[str, float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class PredictionAccuracy:
    """Prediction accuracy metrics"""
    prediction_id: str
    symbol: str
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    directional_accuracy: float  # Percentage of correct direction predictions
    hit_rate: float  # Percentage of predictions within acceptable range
    calculated_at: datetime = None
    
    def __post_init__(self):
        if self.calculated_at is None:
            self.calculated_at = datetime.now()

class PredictionHistoryTracker:
    """
    Comprehensive prediction history tracking and analysis system
    """
    
    def __init__(self, data_storage, db_path: str = "prediction_history.db"):
        """
        Initialize prediction history tracker
        
        Args:
            data_storage: DataStorage instance
            db_path: Path to SQLite database file
        """
        self.data_storage = data_storage
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # In-memory cache for recent predictions
        self.recent_predictions: Dict[str, List[PredictionRecord]] = {}
        self.cache_size = 1000  # Keep last 1000 predictions per symbol
        
        # Performance tracking
        self.accuracy_tracking = {
            'symbols': {},
            'overall_accuracy': 0.0,
            'last_updated': None
        }
        
        logger.info("PredictionHistoryTracker initialized")
    
    def _init_database(self):
        """Initialize SQLite database for prediction history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    prediction_timestamp TEXT NOT NULL,
                    prediction_horizon TEXT NOT NULL,
                    predicted_prices TEXT NOT NULL,
                    prediction_timestamps TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    validation_status TEXT,
                    validation_score REAL,
                    actual_prices TEXT,
                    actual_timestamps TEXT,
                    accuracy_metrics TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create accuracy table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_accuracy (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    mae REAL NOT NULL,
                    mse REAL NOT NULL,
                    rmse REAL NOT NULL,
                    mape REAL NOT NULL,
                    directional_accuracy REAL NOT NULL,
                    hit_rate REAL NOT NULL,
                    calculated_at TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES predictions (prediction_id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON predictions (symbol, prediction_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_created ON predictions (symbol, created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_accuracy_symbol ON prediction_accuracy (symbol)')
            
            conn.commit()
            conn.close()
            
            logger.info("Prediction history database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def store_prediction(self, prediction_result: Dict[str, Any], 
                        validation_result: Optional[Any] = None) -> str:
        """
        Store a prediction result in history
        
        Args:
            prediction_result: Prediction result from PredictionEngine
            validation_result: Optional validation result
            
        Returns:
            Prediction ID
        """
        try:
            # Generate prediction ID
            prediction_id = f"{prediction_result['symbol']}_{prediction_result['prediction_timestamp'].strftime('%Y%m%d_%H%M%S')}"
            
            # Extract prediction data
            predictions_df = prediction_result['predictions']
            predicted_prices = predictions_df['predicted_price'].tolist()
            prediction_timestamps = predictions_df.index.tolist()
            
            # Create prediction record
            record = PredictionRecord(
                prediction_id=prediction_id,
                symbol=prediction_result['symbol'],
                prediction_timestamp=prediction_result['prediction_timestamp'],
                prediction_horizon=prediction_result.get('prediction_horizon', '1 week'),
                predicted_prices=predicted_prices,
                prediction_timestamps=prediction_timestamps,
                confidence_score=prediction_result.get('confidence_score', 0.0),
                model_version=prediction_result.get('model_version', 'unknown'),
                validation_status=validation_result.overall_status.value if validation_result else None,
                validation_score=validation_result.overall_score if validation_result else None
            )
            
            # Store in database
            self._store_prediction_record(record)
            
            # Add to cache
            self._add_to_cache(record)
            
            logger.info(f"Stored prediction {prediction_id} for {record.symbol}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return ""
    
    def _store_prediction_record(self, record: PredictionRecord):
        """Store prediction record in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO predictions (
                    prediction_id, symbol, prediction_timestamp, prediction_horizon,
                    predicted_prices, prediction_timestamps, confidence_score,
                    model_version, validation_status, validation_score,
                    actual_prices, actual_timestamps, accuracy_metrics, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.prediction_id,
                record.symbol,
                record.prediction_timestamp.isoformat(),
                record.prediction_horizon,
                json.dumps(record.predicted_prices),
                json.dumps([ts.isoformat() for ts in record.prediction_timestamps]),
                record.confidence_score,
                record.model_version,
                record.validation_status,
                record.validation_score,
                json.dumps(record.actual_prices) if record.actual_prices else None,
                json.dumps([ts.isoformat() for ts in record.actual_timestamps]) if record.actual_timestamps else None,
                json.dumps(record.accuracy_metrics) if record.accuracy_metrics else None,
                record.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing prediction record: {e}")
            raise
    
    def _add_to_cache(self, record: PredictionRecord):
        """Add prediction record to in-memory cache"""
        try:
            symbol = record.symbol
            if symbol not in self.recent_predictions:
                self.recent_predictions[symbol] = []
            
            self.recent_predictions[symbol].append(record)
            
            # Maintain cache size
            if len(self.recent_predictions[symbol]) > self.cache_size:
                self.recent_predictions[symbol] = self.recent_predictions[symbol][-self.cache_size:]
            
        except Exception as e:
            logger.error(f"Error adding to cache: {e}")
    
    def update_prediction_with_actuals(self, prediction_id: str, 
                                     actual_prices: List[float],
                                     actual_timestamps: List[datetime]) -> bool:
        """
        Update prediction with actual price data
        
        Args:
            prediction_id: Prediction ID
            actual_prices: Actual prices
            actual_timestamps: Actual timestamps
            
        Returns:
            True if successful
        """
        try:
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE predictions 
                SET actual_prices = ?, actual_timestamps = ?
                WHERE prediction_id = ?
            ''', (
                json.dumps(actual_prices),
                json.dumps([ts.isoformat() for ts in actual_timestamps]),
                prediction_id
            ))
            
            conn.commit()
            conn.close()
            
            # Update cache
            for symbol, predictions in self.recent_predictions.items():
                for pred in predictions:
                    if pred.prediction_id == prediction_id:
                        pred.actual_prices = actual_prices
                        pred.actual_timestamps = actual_timestamps
                        break
            
            # Calculate accuracy metrics
            self._calculate_accuracy_metrics(prediction_id, actual_prices, actual_timestamps)
            
            logger.info(f"Updated prediction {prediction_id} with actual data")
            return True
            
        except Exception as e:
            logger.error(f"Error updating prediction with actuals: {e}")
            return False
    
    def _calculate_accuracy_metrics(self, prediction_id: str, 
                                  actual_prices: List[float],
                                  actual_timestamps: List[datetime]):
        """Calculate accuracy metrics for a prediction"""
        try:
            # Get prediction data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, predicted_prices, prediction_timestamps
                FROM predictions WHERE prediction_id = ?
            ''', (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return
            
            symbol, predicted_prices_json, prediction_timestamps_json = result
            predicted_prices = json.loads(predicted_prices_json)
            prediction_timestamps = [datetime.fromisoformat(ts) for ts in json.loads(prediction_timestamps_json)]
            
            conn.close()
            
            # Align predictions and actuals by timestamp
            aligned_data = self._align_predictions_actuals(
                prediction_timestamps, predicted_prices,
                actual_timestamps, actual_prices
            )
            
            if not aligned_data:
                return
            
            pred_aligned, actual_aligned = aligned_data
            
            # Calculate metrics
            mae = np.mean(np.abs(np.array(pred_aligned) - np.array(actual_aligned)))
            mse = np.mean((np.array(pred_aligned) - np.array(actual_aligned)) ** 2)
            rmse = np.sqrt(mse)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((np.array(actual_aligned) - np.array(pred_aligned)) / np.array(actual_aligned))) * 100
            
            # Directional accuracy
            pred_directions = np.diff(pred_aligned) > 0
            actual_directions = np.diff(actual_aligned) > 0
            directional_accuracy = np.mean(pred_directions == actual_directions) * 100
            
            # Hit rate (predictions within 5% of actual)
            hit_rate = np.mean(np.abs((np.array(actual_aligned) - np.array(pred_aligned)) / np.array(actual_aligned)) <= 0.05) * 100
            
            # Store accuracy metrics
            accuracy = PredictionAccuracy(
                prediction_id=prediction_id,
                symbol=symbol,
                mae=mae,
                mse=mse,
                rmse=rmse,
                mape=mape,
                directional_accuracy=directional_accuracy,
                hit_rate=hit_rate
            )
            
            self._store_accuracy_metrics(accuracy)
            
            # Update in-memory accuracy tracking
            self._update_accuracy_tracking(symbol, accuracy)
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
    
    def _align_predictions_actuals(self, pred_timestamps: List[datetime], pred_prices: List[float],
                                 actual_timestamps: List[datetime], actual_prices: List[float]) -> Optional[Tuple[List[float], List[float]]]:
        """Align prediction and actual data by timestamp"""
        try:
            # Create DataFrames for easier alignment
            pred_df = pd.DataFrame({
                'timestamp': pred_timestamps,
                'price': pred_prices
            })
            actual_df = pd.DataFrame({
                'timestamp': actual_timestamps,
                'price': actual_prices
            })
            
            # Merge on timestamp (with tolerance)
            merged = pd.merge_asof(
                pred_df.sort_values('timestamp'),
                actual_df.sort_values('timestamp'),
                on='timestamp',
                tolerance=pd.Timedelta(minutes=30),  # 30-minute tolerance
                direction='nearest'
            )
            
            # Remove rows where no match was found
            merged = merged.dropna()
            
            if len(merged) == 0:
                return None
            
            return merged['price_x'].tolist(), merged['price_y'].tolist()
            
        except Exception as e:
            logger.error(f"Error aligning predictions and actuals: {e}")
            return None
    
    def _store_accuracy_metrics(self, accuracy: PredictionAccuracy):
        """Store accuracy metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO prediction_accuracy (
                    prediction_id, symbol, mae, mse, rmse, mape,
                    directional_accuracy, hit_rate, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                accuracy.prediction_id,
                accuracy.symbol,
                accuracy.mae,
                accuracy.mse,
                accuracy.rmse,
                accuracy.mape,
                accuracy.directional_accuracy,
                accuracy.hit_rate,
                accuracy.calculated_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing accuracy metrics: {e}")
    
    def _update_accuracy_tracking(self, symbol: str, accuracy: PredictionAccuracy):
        """Update in-memory accuracy tracking"""
        try:
            if symbol not in self.accuracy_tracking['symbols']:
                self.accuracy_tracking['symbols'][symbol] = {
                    'recent_accuracies': [],
                    'average_accuracy': 0.0,
                    'last_updated': None
                }
            
            symbol_data = self.accuracy_tracking['symbols'][symbol]
            symbol_data['recent_accuracies'].append(accuracy.hit_rate)
            
            # Keep only last 100 accuracies
            if len(symbol_data['recent_accuracies']) > 100:
                symbol_data['recent_accuracies'] = symbol_data['recent_accuracies'][-100:]
            
            # Update average accuracy
            symbol_data['average_accuracy'] = np.mean(symbol_data['recent_accuracies'])
            symbol_data['last_updated'] = datetime.now()
            
            # Update overall accuracy
            all_accuracies = []
            for sym_data in self.accuracy_tracking['symbols'].values():
                all_accuracies.extend(sym_data['recent_accuracies'])
            
            if all_accuracies:
                self.accuracy_tracking['overall_accuracy'] = np.mean(all_accuracies)
                self.accuracy_tracking['last_updated'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating accuracy tracking: {e}")
    
    def get_prediction_history(self, symbol: str, days_back: int = 30) -> List[PredictionRecord]:
        """
        Get prediction history for a symbol
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            List of PredictionRecord objects
        """
        try:
            # Try cache first
            if symbol in self.recent_predictions:
                cutoff_time = datetime.now() - timedelta(days=days_back)
                cached_predictions = [
                    pred for pred in self.recent_predictions[symbol]
                    if pred.created_at >= cutoff_time
                ]
                if len(cached_predictions) > 0:
                    return cached_predictions
            
            # Query database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(days=days_back)
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE symbol = ? AND created_at >= ?
                ORDER BY created_at DESC
            ''', (symbol, cutoff_time.isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            # Convert to PredictionRecord objects
            predictions = []
            for row in results:
                record = self._row_to_prediction_record(row)
                predictions.append(record)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting prediction history for {symbol}: {e}")
            return []
    
    def _row_to_prediction_record(self, row: Tuple) -> PredictionRecord:
        """Convert database row to PredictionRecord"""
        return PredictionRecord(
            prediction_id=row[0],
            symbol=row[1],
            prediction_timestamp=datetime.fromisoformat(row[2]),
            prediction_horizon=row[3],
            predicted_prices=json.loads(row[4]),
            prediction_timestamps=[datetime.fromisoformat(ts) for ts in json.loads(row[5])],
            confidence_score=row[6],
            model_version=row[7],
            validation_status=row[8],
            validation_score=row[9],
            actual_prices=json.loads(row[10]) if row[10] else None,
            actual_timestamps=[datetime.fromisoformat(ts) for ts in json.loads(row[11])] if row[11] else None,
            accuracy_metrics=json.loads(row[12]) if row[12] else None,
            created_at=datetime.fromisoformat(row[13])
        )
    
    def get_accuracy_statistics(self, symbol: Optional[str] = None, days_back: int = 30) -> Dict[str, Any]:
        """
        Get accuracy statistics
        
        Args:
            symbol: Optional symbol to filter by
            days_back: Number of days to look back
            
        Returns:
            Dictionary with accuracy statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            if symbol:
                cursor.execute('''
                    SELECT * FROM prediction_accuracy 
                    WHERE symbol = ? AND calculated_at >= ?
                    ORDER BY calculated_at DESC
                ''', (symbol, cutoff_time.isoformat()))
            else:
                cursor.execute('''
                    SELECT * FROM prediction_accuracy 
                    WHERE calculated_at >= ?
                    ORDER BY calculated_at DESC
                ''', (cutoff_time.isoformat(),))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {'error': 'No accuracy data available'}
            
            # Calculate statistics
            mae_values = [row[2] for row in results]
            mse_values = [row[3] for row in results]
            rmse_values = [row[4] for row in results]
            mape_values = [row[5] for row in results]
            directional_accuracies = [row[6] for row in results]
            hit_rates = [row[7] for row in results]
            
            return {
                'total_predictions': len(results),
                'mae': {
                    'mean': np.mean(mae_values),
                    'std': np.std(mae_values),
                    'min': np.min(mae_values),
                    'max': np.max(mae_values)
                },
                'mse': {
                    'mean': np.mean(mse_values),
                    'std': np.std(mse_values),
                    'min': np.min(mse_values),
                    'max': np.max(mse_values)
                },
                'rmse': {
                    'mean': np.mean(rmse_values),
                    'std': np.std(rmse_values),
                    'min': np.min(rmse_values),
                    'max': np.max(rmse_values)
                },
                'mape': {
                    'mean': np.mean(mape_values),
                    'std': np.std(mape_values),
                    'min': np.min(mape_values),
                    'max': np.max(mape_values)
                },
                'directional_accuracy': {
                    'mean': np.mean(directional_accuracies),
                    'std': np.std(directional_accuracies),
                    'min': np.min(directional_accuracies),
                    'max': np.max(directional_accuracies)
                },
                'hit_rate': {
                    'mean': np.mean(hit_rates),
                    'std': np.std(hit_rates),
                    'min': np.min(hit_rates),
                    'max': np.max(hit_rates)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting accuracy statistics: {e}")
            return {'error': str(e)}
    
    def get_performance_trends(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get performance trends over time
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            Dictionary with performance trends
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(days=days_back)
            cursor.execute('''
                SELECT calculated_at, hit_rate, directional_accuracy, mape
                FROM prediction_accuracy 
                WHERE symbol = ? AND calculated_at >= ?
                ORDER BY calculated_at ASC
            ''', (symbol, cutoff_time.isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {'error': 'No performance data available'}
            
            # Create DataFrame for trend analysis
            df = pd.DataFrame(results, columns=['timestamp', 'hit_rate', 'directional_accuracy', 'mape'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate trends
            hit_rate_trend = np.polyfit(range(len(df)), df['hit_rate'], 1)[0]
            directional_trend = np.polyfit(range(len(df)), df['directional_accuracy'], 1)[0]
            mape_trend = np.polyfit(range(len(df)), df['mape'], 1)[0]
            
            return {
                'total_predictions': len(df),
                'hit_rate_trend': hit_rate_trend,
                'directional_accuracy_trend': directional_trend,
                'mape_trend': mape_trend,
                'recent_performance': {
                    'hit_rate': df['hit_rate'].tail(10).mean(),
                    'directional_accuracy': df['directional_accuracy'].tail(10).mean(),
                    'mape': df['mape'].tail(10).mean()
                },
                'overall_performance': {
                    'hit_rate': df['hit_rate'].mean(),
                    'directional_accuracy': df['directional_accuracy'].mean(),
                    'mape': df['mape'].mean()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance trends for {symbol}: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, days_back: int = 90):
        """Clean up old prediction data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old predictions
            cursor.execute('DELETE FROM predictions WHERE created_at < ?', (cutoff_time.isoformat(),))
            deleted_predictions = cursor.rowcount
            
            # Delete old accuracy data
            cursor.execute('DELETE FROM prediction_accuracy WHERE calculated_at < ?', (cutoff_time.isoformat(),))
            deleted_accuracy = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            # Clean up cache
            for symbol in list(self.recent_predictions.keys()):
                self.recent_predictions[symbol] = [
                    pred for pred in self.recent_predictions[symbol]
                    if pred.created_at >= cutoff_time
                ]
                if not self.recent_predictions[symbol]:
                    del self.recent_predictions[symbol]
            
            logger.info(f"Cleaned up {deleted_predictions} old predictions and {deleted_accuracy} accuracy records")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("PredictionHistoryTracker module loaded successfully")
    print("Use with DataStorage instance for full functionality")
