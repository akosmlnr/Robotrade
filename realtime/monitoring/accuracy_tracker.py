"""
Prediction Accuracy Tracking for Real-time LSTM Prediction System
Phase 3.3: Comprehensive accuracy tracking and analysis over time
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

class AccuracyMetric(Enum):
    """Types of accuracy metrics"""
    MAE = "mae"  # Mean Absolute Error
    MSE = "mse"  # Mean Squared Error
    RMSE = "rmse"  # Root Mean Squared Error
    MAPE = "mape"  # Mean Absolute Percentage Error
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    HIT_RATE = "hit_rate"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"

class TimeFrame(Enum):
    """Time frames for accuracy analysis"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class AccuracyRecord:
    """Record of prediction accuracy"""
    symbol: str
    prediction_timestamp: datetime
    actual_timestamp: datetime
    predicted_price: float
    actual_price: float
    mae: float
    mse: float
    rmse: float
    mape: float
    directional_accuracy: float
    hit_rate: float
    confidence_score: float
    model_version: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class AccuracySummary:
    """Summary of accuracy metrics over a time period"""
    symbol: str
    time_frame: TimeFrame
    start_date: datetime
    end_date: datetime
    total_predictions: int
    mae_mean: float
    mae_std: float
    mse_mean: float
    mse_std: float
    rmse_mean: float
    rmse_std: float
    mape_mean: float
    mape_std: float
    directional_accuracy_mean: float
    directional_accuracy_std: float
    hit_rate_mean: float
    hit_rate_std: float
    confidence_score_mean: float
    confidence_score_std: float
    trend_direction: str  # "improving", "declining", "stable"
    calculated_at: datetime = None
    
    def __post_init__(self):
        if self.calculated_at is None:
            self.calculated_at = datetime.now()

class AccuracyTracker:
    """
    Comprehensive prediction accuracy tracking and analysis system
    """
    
    def __init__(self, data_storage, config: Dict[str, Any] = None):
        """
        Initialize accuracy tracker
        
        Args:
            data_storage: DataStorage instance
            config: Optional configuration dictionary
        """
        self.data_storage = data_storage
        
        # Configuration
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # Accuracy tracking settings
        self.accuracy_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'poor': 0.6,
            'unacceptable': 0.5
        }
        
        # Accuracy history cache
        self.accuracy_cache: Dict[str, List[AccuracyRecord]] = {}
        self.cache_size = 1000  # Keep last 1000 records per symbol
        
        # Accuracy statistics
        self.accuracy_stats = {
            'total_predictions_tracked': 0,
            'symbols_tracked': set(),
            'overall_accuracy': 0.0,
            'last_updated': None
        }
        
        # Performance tracking
        self.performance_trends: Dict[str, Dict[str, List[float]]] = {}
        
        logger.info("AccuracyTracker initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'accuracy_tracking_enabled': True,
            'real_time_tracking': True,
            'accuracy_calculation_interval_minutes': 15,
            'trend_analysis_window_days': 30,
            'performance_alert_threshold': 0.1,  # 10% performance degradation
            'min_predictions_for_analysis': 10,
            'accuracy_visualization_enabled': True,
            'export_accuracy_reports': True
        }
    
    def track_prediction_accuracy(self, symbol: str, prediction_timestamp: datetime,
                                predicted_price: float, actual_price: float,
                                confidence_score: float, model_version: str = "unknown") -> bool:
        """
        Track the accuracy of a prediction
        
        Args:
            symbol: Stock symbol
            prediction_timestamp: When the prediction was made
            predicted_price: Predicted price
            actual_price: Actual price
            confidence_score: Confidence score of the prediction
            model_version: Version of the model used
            
        Returns:
            True if tracking was successful
        """
        try:
            # Calculate accuracy metrics
            mae = abs(predicted_price - actual_price)
            mse = (predicted_price - actual_price) ** 2
            rmse = np.sqrt(mse)
            mape = abs((actual_price - predicted_price) / actual_price) * 100 if actual_price != 0 else 0
            
            # Directional accuracy (simplified - assumes we're predicting next price direction)
            # In practice, you'd need to compare with previous price
            directional_accuracy = 1.0 if (predicted_price > actual_price * 0.99 and predicted_price < actual_price * 1.01) else 0.0
            
            # Hit rate (within 5% of actual price)
            hit_rate = 1.0 if mape <= 5.0 else 0.0
            
            # Create accuracy record
            accuracy_record = AccuracyRecord(
                symbol=symbol,
                prediction_timestamp=prediction_timestamp,
                actual_timestamp=datetime.now(),
                predicted_price=predicted_price,
                actual_price=actual_price,
                mae=mae,
                mse=mse,
                rmse=rmse,
                mape=mape,
                directional_accuracy=directional_accuracy,
                hit_rate=hit_rate,
                confidence_score=confidence_score,
                model_version=model_version
            )
            
            # Store in database
            self._store_accuracy_record(accuracy_record)
            
            # Add to cache
            self._add_to_cache(accuracy_record)
            
            # Update statistics
            self._update_accuracy_stats(accuracy_record)
            
            # Check for performance alerts
            self._check_performance_alerts(symbol, accuracy_record)
            
            logger.debug(f"Tracked accuracy for {symbol}: MAE={mae:.4f}, MAPE={mape:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking prediction accuracy for {symbol}: {e}")
            return False
    
    def _store_accuracy_record(self, record: AccuracyRecord):
        """Store accuracy record in database"""
        try:
            cursor = self.data_storage.connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO prediction_accuracy 
                (symbol, prediction_timestamp, actual_timestamp, predicted_price, actual_price,
                 mae, mse, rmse, mape, directional_accuracy, hit_rate, confidence_score, 
                 model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.symbol, record.prediction_timestamp, record.actual_timestamp,
                record.predicted_price, record.actual_price, record.mae, record.mse,
                record.rmse, record.mape, record.directional_accuracy, record.hit_rate,
                record.confidence_score, record.model_version, record.created_at
            ))
            
            self.data_storage.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing accuracy record: {e}")
    
    def _add_to_cache(self, record: AccuracyRecord):
        """Add accuracy record to cache"""
        try:
            symbol = record.symbol
            if symbol not in self.accuracy_cache:
                self.accuracy_cache[symbol] = []
            
            self.accuracy_cache[symbol].append(record)
            
            # Maintain cache size
            if len(self.accuracy_cache[symbol]) > self.cache_size:
                self.accuracy_cache[symbol] = self.accuracy_cache[symbol][-self.cache_size:]
            
        except Exception as e:
            logger.error(f"Error adding to accuracy cache: {e}")
    
    def _update_accuracy_stats(self, record: AccuracyRecord):
        """Update accuracy statistics"""
        try:
            self.accuracy_stats['total_predictions_tracked'] += 1
            self.accuracy_stats['symbols_tracked'].add(record.symbol)
            self.accuracy_stats['last_updated'] = datetime.now()
            
            # Update overall accuracy (using hit rate as primary metric)
            if self.accuracy_stats['total_predictions_tracked'] == 1:
                self.accuracy_stats['overall_accuracy'] = record.hit_rate
            else:
                # Exponential moving average
                alpha = 0.1
                self.accuracy_stats['overall_accuracy'] = (
                    alpha * record.hit_rate + 
                    (1 - alpha) * self.accuracy_stats['overall_accuracy']
                )
            
        except Exception as e:
            logger.error(f"Error updating accuracy stats: {e}")
    
    def _check_performance_alerts(self, symbol: str, record: AccuracyRecord):
        """Check for performance degradation alerts"""
        try:
            # Get recent performance for symbol
            recent_accuracy = self.get_symbol_accuracy_summary(
                symbol, days_back=7, time_frame=TimeFrame.DAILY
            )
            
            if not recent_accuracy:
                return
            
            # Check if performance has degraded significantly
            current_hit_rate = record.hit_rate
            recent_hit_rate = recent_accuracy.hit_rate_mean
            
            if recent_hit_rate - current_hit_rate > self.config.get('performance_alert_threshold', 0.1):
                # Send alert
                alert_message = f"Performance degradation detected for {symbol}: " \
                              f"Current hit rate {current_hit_rate:.2f}, " \
                              f"Recent average {recent_hit_rate:.2f}"
                
                self.data_storage.store_system_alert(
                    alert_type="PERFORMANCE_DEGRADATION",
                    severity="WARNING",
                    message=alert_message,
                    details={
                        'symbol': symbol,
                        'current_hit_rate': current_hit_rate,
                        'recent_hit_rate': recent_hit_rate,
                        'degradation': recent_hit_rate - current_hit_rate
                    }
                )
                
                logger.warning(alert_message)
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def get_symbol_accuracy_summary(self, symbol: str, days_back: int = 30,
                                   time_frame: TimeFrame = TimeFrame.DAILY) -> Optional[AccuracySummary]:
        """
        Get accuracy summary for a symbol over a time period
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            time_frame: Time frame for analysis
            
        Returns:
            AccuracySummary object
        """
        try:
            # Get accuracy records
            records = self._get_accuracy_records(symbol, days_back)
            
            if len(records) < self.config.get('min_predictions_for_analysis', 10):
                return None
            
            # Calculate summary statistics
            mae_values = [r.mae for r in records]
            mse_values = [r.mse for r in records]
            rmse_values = [r.rmse for r in records]
            mape_values = [r.mape for r in records]
            directional_values = [r.directional_accuracy for r in records]
            hit_rate_values = [r.hit_rate for r in records]
            confidence_values = [r.confidence_score for r in records]
            
            # Calculate trend direction
            trend_direction = self._calculate_trend_direction(hit_rate_values)
            
            # Create summary
            summary = AccuracySummary(
                symbol=symbol,
                time_frame=time_frame,
                start_date=records[0].prediction_timestamp,
                end_date=records[-1].prediction_timestamp,
                total_predictions=len(records),
                mae_mean=np.mean(mae_values),
                mae_std=np.std(mae_values),
                mse_mean=np.mean(mse_values),
                mse_std=np.std(mse_values),
                rmse_mean=np.mean(rmse_values),
                rmse_std=np.std(rmse_values),
                mape_mean=np.mean(mape_values),
                mape_std=np.std(mape_values),
                directional_accuracy_mean=np.mean(directional_values),
                directional_accuracy_std=np.std(directional_values),
                hit_rate_mean=np.mean(hit_rate_values),
                hit_rate_std=np.std(hit_rate_values),
                confidence_score_mean=np.mean(confidence_values),
                confidence_score_std=np.std(confidence_values),
                trend_direction=trend_direction
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting accuracy summary for {symbol}: {e}")
            return None
    
    def _get_accuracy_records(self, symbol: str, days_back: int) -> List[AccuracyRecord]:
        """Get accuracy records for a symbol"""
        try:
            # Try cache first
            if symbol in self.accuracy_cache:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                cached_records = [
                    record for record in self.accuracy_cache[symbol]
                    if record.prediction_timestamp >= cutoff_date
                ]
                if len(cached_records) > 0:
                    return cached_records
            
            # Query database
            cursor = self.data_storage.connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            cursor.execute("""
                SELECT * FROM prediction_accuracy 
                WHERE symbol = ? AND prediction_timestamp >= ?
                ORDER BY prediction_timestamp ASC
            """, (symbol, cutoff_date))
            
            rows = cursor.fetchall()
            
            # Convert to AccuracyRecord objects
            records = []
            for row in rows:
                record = AccuracyRecord(
                    symbol=row[1],
                    prediction_timestamp=datetime.fromisoformat(row[2]),
                    actual_timestamp=datetime.fromisoformat(row[3]),
                    predicted_price=row[4],
                    actual_price=row[5],
                    mae=row[6],
                    mse=row[7],
                    rmse=row[8],
                    mape=row[9],
                    directional_accuracy=row[10],
                    hit_rate=row[11],
                    confidence_score=row[12],
                    model_version=row[13],
                    created_at=datetime.fromisoformat(row[14])
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Error getting accuracy records for {symbol}: {e}")
            return []
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        try:
            if len(values) < 3:
                return "insufficient_data"
            
            # Use linear regression to determine trend
            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, values)
            
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
            else:
                return "stable"
            
        except Exception as e:
            logger.error(f"Error calculating trend direction: {e}")
            return "unknown"
    
    def get_overall_accuracy_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get overall accuracy summary across all symbols
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dictionary with overall accuracy summary
        """
        try:
            # Get all symbols
            symbols = list(self.accuracy_stats['symbols_tracked'])
            
            if not symbols:
                return {'error': 'No symbols tracked'}
            
            # Calculate overall metrics
            all_mae = []
            all_mse = []
            all_rmse = []
            all_mape = []
            all_directional = []
            all_hit_rates = []
            all_confidence = []
            
            symbol_summaries = {}
            
            for symbol in symbols:
                summary = self.get_symbol_accuracy_summary(symbol, days_back)
                if summary:
                    symbol_summaries[symbol] = {
                        'hit_rate': summary.hit_rate_mean,
                        'mape': summary.mape_mean,
                        'directional_accuracy': summary.directional_accuracy_mean,
                        'total_predictions': summary.total_predictions,
                        'trend': summary.trend_direction
                    }
                    
                    # Add to overall metrics
                    all_hit_rates.append(summary.hit_rate_mean)
                    all_mape.append(summary.mape_mean)
                    all_directional.append(summary.directional_accuracy_mean)
            
            # Calculate overall statistics
            overall_summary = {
                'total_symbols': len(symbols),
                'total_predictions': sum(s['total_predictions'] for s in symbol_summaries.values()),
                'overall_hit_rate': np.mean(all_hit_rates) if all_hit_rates else 0,
                'overall_mape': np.mean(all_mape) if all_mape else 0,
                'overall_directional_accuracy': np.mean(all_directional) if all_directional else 0,
                'hit_rate_std': np.std(all_hit_rates) if all_hit_rates else 0,
                'mape_std': np.std(all_mape) if all_mape else 0,
                'symbol_summaries': symbol_summaries,
                'performance_distribution': {
                    'excellent': len([s for s in symbol_summaries.values() if s['hit_rate'] >= self.accuracy_thresholds['excellent']]),
                    'good': len([s for s in symbol_summaries.values() if self.accuracy_thresholds['good'] <= s['hit_rate'] < self.accuracy_thresholds['excellent']]),
                    'acceptable': len([s for s in symbol_summaries.values() if self.accuracy_thresholds['acceptable'] <= s['hit_rate'] < self.accuracy_thresholds['good']]),
                    'poor': len([s for s in symbol_summaries.values() if self.accuracy_thresholds['poor'] <= s['hit_rate'] < self.accuracy_thresholds['acceptable']]),
                    'unacceptable': len([s for s in symbol_summaries.values() if s['hit_rate'] < self.accuracy_thresholds['poor']])
                },
                'trend_distribution': {
                    'improving': len([s for s in symbol_summaries.values() if s['trend'] == 'improving']),
                    'stable': len([s for s in symbol_summaries.values() if s['trend'] == 'stable']),
                    'declining': len([s for s in symbol_summaries.values() if s['trend'] == 'declining'])
                }
            }
            
            return overall_summary
            
        except Exception as e:
            logger.error(f"Error getting overall accuracy summary: {e}")
            return {'error': str(e)}
    
    def generate_accuracy_report(self, symbol: str = None, days_back: int = 30,
                               output_path: str = None) -> str:
        """
        Generate accuracy report with visualizations
        
        Args:
            symbol: Specific symbol (None for all symbols)
            days_back: Number of days to analyze
            output_path: Output path for report
            
        Returns:
            Path to generated report
        """
        try:
            if not self.config.get('accuracy_visualization_enabled', True):
                return ""
            
            # Set up output path
            if not output_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"accuracy_report_{timestamp}.html"
            
            # Generate report content
            if symbol:
                report_content = self._generate_symbol_report(symbol, days_back)
            else:
                report_content = self._generate_overall_report(days_back)
            
            # Save report
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Accuracy report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating accuracy report: {e}")
            return ""
    
    def _generate_symbol_report(self, symbol: str, days_back: int) -> str:
        """Generate accuracy report for a specific symbol"""
        try:
            # Get accuracy summary
            summary = self.get_symbol_accuracy_summary(symbol, days_back)
            if not summary:
                return f"<html><body><h1>Accuracy Report for {symbol}</h1><p>Insufficient data for analysis</p></body></html>"
            
            # Get detailed records
            records = self._get_accuracy_records(symbol, days_back)
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Accuracy Analysis for {symbol}', fontsize=16)
            
            # Hit rate over time
            timestamps = [r.prediction_timestamp for r in records]
            hit_rates = [r.hit_rate for r in records]
            axes[0, 0].plot(timestamps, hit_rates, marker='o', markersize=3)
            axes[0, 0].set_title('Hit Rate Over Time')
            axes[0, 0].set_ylabel('Hit Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # MAPE over time
            mape_values = [r.mape for r in records]
            axes[0, 1].plot(timestamps, mape_values, marker='o', markersize=3, color='red')
            axes[0, 1].set_title('MAPE Over Time')
            axes[0, 1].set_ylabel('MAPE (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Accuracy distribution
            axes[1, 0].hist(hit_rates, bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Hit Rate Distribution')
            axes[1, 0].set_xlabel('Hit Rate')
            axes[1, 0].set_ylabel('Frequency')
            
            # Confidence vs Accuracy
            confidence_scores = [r.confidence_score for r in records]
            axes[1, 1].scatter(confidence_scores, hit_rates, alpha=0.6)
            axes[1, 1].set_title('Confidence vs Hit Rate')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Hit Rate')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"accuracy_plot_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate HTML report
            html_content = f"""
            <html>
            <head>
                <title>Accuracy Report for {symbol}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                    .metric {{ margin: 10px 0; }}
                    .trend {{ font-weight: bold; }}
                    .improving {{ color: green; }}
                    .declining {{ color: red; }}
                    .stable {{ color: blue; }}
                </style>
            </head>
            <body>
                <h1>Accuracy Report for {symbol}</h1>
                <div class="summary">
                    <h2>Summary Statistics</h2>
                    <div class="metric">Total Predictions: {summary.total_predictions}</div>
                    <div class="metric">Time Period: {summary.start_date.date()} to {summary.end_date.date()}</div>
                    <div class="metric">Hit Rate: {summary.hit_rate_mean:.3f} ± {summary.hit_rate_std:.3f}</div>
                    <div class="metric">MAPE: {summary.mape_mean:.2f}% ± {summary.mape_std:.2f}%</div>
                    <div class="metric">Directional Accuracy: {summary.directional_accuracy_mean:.3f} ± {summary.directional_accuracy_std:.3f}</div>
                    <div class="metric">Confidence Score: {summary.confidence_score_mean:.3f} ± {summary.confidence_score_std:.3f}</div>
                    <div class="metric">Trend: <span class="trend {summary.trend_direction}">{summary.trend_direction.title()}</span></div>
                </div>
                <h2>Visualizations</h2>
                <img src="{plot_path}" alt="Accuracy Analysis" style="max-width: 100%;">
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating symbol report: {e}")
            return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
    
    def _generate_overall_report(self, days_back: int) -> str:
        """Generate overall accuracy report"""
        try:
            # Get overall summary
            overall_summary = self.get_overall_accuracy_summary(days_back)
            
            if 'error' in overall_summary:
                return f"<html><body><h1>Overall Accuracy Report</h1><p>{overall_summary['error']}</p></body></html>"
            
            # Generate HTML report
            html_content = f"""
            <html>
            <head>
                <title>Overall Accuracy Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                    .metric {{ margin: 10px 0; }}
                    .symbol-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    .symbol-table th, .symbol-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .symbol-table th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Overall Accuracy Report</h1>
                <div class="summary">
                    <h2>Overall Statistics</h2>
                    <div class="metric">Total Symbols: {overall_summary['total_symbols']}</div>
                    <div class="metric">Total Predictions: {overall_summary['total_predictions']}</div>
                    <div class="metric">Overall Hit Rate: {overall_summary['overall_hit_rate']:.3f} ± {overall_summary['hit_rate_std']:.3f}</div>
                    <div class="metric">Overall MAPE: {overall_summary['overall_mape']:.2f}% ± {overall_summary['mape_std']:.2f}%</div>
                    <div class="metric">Overall Directional Accuracy: {overall_summary['overall_directional_accuracy']:.3f}</div>
                </div>
                
                <h2>Performance Distribution</h2>
                <div class="summary">
                    <div class="metric">Excellent (≥90%): {overall_summary['performance_distribution']['excellent']} symbols</div>
                    <div class="metric">Good (80-90%): {overall_summary['performance_distribution']['good']} symbols</div>
                    <div class="metric">Acceptable (70-80%): {overall_summary['performance_distribution']['acceptable']} symbols</div>
                    <div class="metric">Poor (60-70%): {overall_summary['performance_distribution']['poor']} symbols</div>
                    <div class="metric">Unacceptable (<60%): {overall_summary['performance_distribution']['unacceptable']} symbols</div>
                </div>
                
                <h2>Trend Distribution</h2>
                <div class="summary">
                    <div class="metric">Improving: {overall_summary['trend_distribution']['improving']} symbols</div>
                    <div class="metric">Stable: {overall_summary['trend_distribution']['stable']} symbols</div>
                    <div class="metric">Declining: {overall_summary['trend_distribution']['declining']} symbols</div>
                </div>
                
                <h2>Symbol Performance</h2>
                <table class="symbol-table">
                    <tr>
                        <th>Symbol</th>
                        <th>Hit Rate</th>
                        <th>MAPE</th>
                        <th>Directional Accuracy</th>
                        <th>Predictions</th>
                        <th>Trend</th>
                    </tr>
            """
            
            # Add symbol rows
            for symbol, data in overall_summary['symbol_summaries'].items():
                html_content += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{data['hit_rate']:.3f}</td>
                        <td>{data['mape']:.2f}%</td>
                        <td>{data['directional_accuracy']:.3f}</td>
                        <td>{data['total_predictions']}</td>
                        <td>{data['trend'].title()}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating overall report: {e}")
            return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
    
    def get_accuracy_statistics(self) -> Dict[str, Any]:
        """Get accuracy tracking statistics"""
        try:
            return {
                'accuracy_stats': self.accuracy_stats,
                'symbols_tracked': list(self.accuracy_stats['symbols_tracked']),
                'total_symbols': len(self.accuracy_stats['symbols_tracked']),
                'overall_accuracy': self.accuracy_stats['overall_accuracy'],
                'last_updated': self.accuracy_stats['last_updated'].isoformat() if self.accuracy_stats['last_updated'] else None,
                'accuracy_thresholds': self.accuracy_thresholds,
                'config': self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting accuracy statistics: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("AccuracyTracker module loaded successfully")
    print("Use with DataStorage instance for full functionality")
