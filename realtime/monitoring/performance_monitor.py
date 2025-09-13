"""
Performance Monitoring for Real-time LSTM Prediction System
Phase 3.3: Comprehensive model and system performance monitoring
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import psutil
import threading
import time
from collections import deque
import json

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    SYSTEM = "system"
    MODEL = "model"
    PREDICTION = "prediction"
    VALIDATION = "validation"
    DATA_QUALITY = "data_quality"
    BUSINESS = "business"

class MetricCategory(Enum):
    """Categories of metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"

@dataclass
class PerformanceMetric:
    """Performance metric record"""
    metric_name: str
    metric_type: MetricType
    metric_category: MetricCategory
    value: float
    unit: str
    timestamp: datetime
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    metric_name: str
    alert_type: str
    severity: str
    threshold_value: float
    actual_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for models and infrastructure
    """
    
    def __init__(self, data_storage, config: Dict[str, Any] = None):
        """
        Initialize performance monitor
        
        Args:
            data_storage: DataStorage instance
            config: Optional configuration dictionary
        """
        self.data_storage = data_storage
        
        # Configuration
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # Performance thresholds
        self.thresholds = self.config.get('thresholds', {})
        
        # Metric storage
        self.metric_buffer: Dict[str, deque] = {}
        self.buffer_size = self.config.get('buffer_size', 1000)
        
        # Performance alerts
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        
        # Monitoring threads
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.stop_monitoring = threading.Event()
        
        # Callback functions
        self.on_metric_threshold_exceeded: Optional[Callable] = None
        
        # Performance statistics
        self.performance_stats = {
            'total_metrics_collected': 0,
            'active_alerts': 0,
            'last_updated': None,
            'system_uptime': datetime.now()
        }
        
        # Start monitoring if enabled
        if self.config.get('auto_monitoring_enabled', True):
            self._start_monitoring()
        
        logger.info("PerformanceMonitor initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'auto_monitoring_enabled': True,
            'monitoring_interval_seconds': 60,
            'buffer_size': 1000,
            'alert_cooldown_minutes': 5,
            'thresholds': {
                'prediction_latency_ms': 5000,  # 5 seconds
                'prediction_throughput_per_minute': 10,
                'model_accuracy': 0.7,
                'cpu_usage_percent': 80,
                'memory_usage_percent': 85,
                'disk_usage_percent': 90,
                'error_rate_percent': 5,
                'validation_failure_rate': 0.1
            },
            'monitored_metrics': [
                'prediction_latency_ms',
                'prediction_throughput_per_minute',
                'model_accuracy',
                'cpu_usage_percent',
                'memory_usage_percent',
                'disk_usage_percent',
                'error_rate_percent',
                'validation_failure_rate',
                'data_freshness_minutes',
                'prediction_confidence_score'
            ]
        }
    
    def _start_monitoring(self):
        """Start automatic monitoring"""
        try:
            # Start system metrics monitoring
            self._start_system_monitoring()
            
            # Start model metrics monitoring
            self._start_model_monitoring()
            
            # Start prediction metrics monitoring
            self._start_prediction_monitoring()
            
            logger.info("Performance monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting performance monitoring: {e}")
    
    def _start_system_monitoring(self):
        """Start system metrics monitoring thread"""
        try:
            thread = threading.Thread(
                target=self._monitor_system_metrics,
                daemon=True,
                name="SystemMonitor"
            )
            thread.start()
            self.monitoring_threads['system'] = thread
            
        except Exception as e:
            logger.error(f"Error starting system monitoring: {e}")
    
    def _start_model_monitoring(self):
        """Start model metrics monitoring thread"""
        try:
            thread = threading.Thread(
                target=self._monitor_model_metrics,
                daemon=True,
                name="ModelMonitor"
            )
            thread.start()
            self.monitoring_threads['model'] = thread
            
        except Exception as e:
            logger.error(f"Error starting model monitoring: {e}")
    
    def _start_prediction_monitoring(self):
        """Start prediction metrics monitoring thread"""
        try:
            thread = threading.Thread(
                target=self._monitor_prediction_metrics,
                daemon=True,
                name="PredictionMonitor"
            )
            thread.start()
            self.monitoring_threads['prediction'] = thread
            
        except Exception as e:
            logger.error(f"Error starting prediction monitoring: {e}")
    
    def _monitor_system_metrics(self):
        """Monitor system performance metrics"""
        while not self.stop_monitoring.is_set():
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self._record_metric(
                    metric_name='cpu_usage_percent',
                    metric_type=MetricType.SYSTEM,
                    metric_category=MetricCategory.RESOURCE_USAGE,
                    value=cpu_percent,
                    unit='percent'
                )
                
                # Memory usage
                memory = psutil.virtual_memory()
                self._record_metric(
                    metric_name='memory_usage_percent',
                    metric_type=MetricType.SYSTEM,
                    metric_category=MetricCategory.RESOURCE_USAGE,
                    value=memory.percent,
                    unit='percent'
                )
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self._record_metric(
                    metric_name='disk_usage_percent',
                    metric_type=MetricType.SYSTEM,
                    metric_category=MetricCategory.RESOURCE_USAGE,
                    value=disk_percent,
                    unit='percent'
                )
                
                # Network I/O
                network = psutil.net_io_counters()
                self._record_metric(
                    metric_name='network_bytes_sent',
                    metric_type=MetricType.SYSTEM,
                    metric_category=MetricCategory.THROUGHPUT,
                    value=network.bytes_sent,
                    unit='bytes'
                )
                
                self._record_metric(
                    metric_name='network_bytes_recv',
                    metric_type=MetricType.SYSTEM,
                    metric_category=MetricCategory.THROUGHPUT,
                    value=network.bytes_recv,
                    unit='bytes'
                )
                
                # Wait for next monitoring cycle
                self.stop_monitoring.wait(self.config.get('monitoring_interval_seconds', 60))
                
            except Exception as e:
                logger.error(f"Error in system metrics monitoring: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _monitor_model_metrics(self):
        """Monitor model performance metrics"""
        while not self.stop_monitoring.is_set():
            try:
                # Get recent model performance from database
                recent_accuracy = self._get_recent_model_accuracy()
                if recent_accuracy is not None:
                    self._record_metric(
                        metric_name='model_accuracy',
                        metric_type=MetricType.MODEL,
                        metric_category=MetricCategory.ACCURACY,
                        value=recent_accuracy,
                        unit='ratio'
                    )
                
                # Get model prediction count
                prediction_count = self._get_prediction_count_last_hour()
                self._record_metric(
                    metric_name='predictions_per_hour',
                    metric_type=MetricType.MODEL,
                    metric_category=MetricCategory.THROUGHPUT,
                    value=prediction_count,
                    unit='count'
                )
                
                # Wait for next monitoring cycle
                self.stop_monitoring.wait(self.config.get('monitoring_interval_seconds', 60))
                
            except Exception as e:
                logger.error(f"Error in model metrics monitoring: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _monitor_prediction_metrics(self):
        """Monitor prediction performance metrics"""
        while not self.stop_monitoring.is_set():
            try:
                # Get prediction latency
                avg_latency = self._get_average_prediction_latency()
                if avg_latency is not None:
                    self._record_metric(
                        metric_name='prediction_latency_ms',
                        metric_type=MetricType.PREDICTION,
                        metric_category=MetricCategory.LATENCY,
                        value=avg_latency,
                        unit='milliseconds'
                    )
                
                # Get validation failure rate
                validation_failure_rate = self._get_validation_failure_rate()
                if validation_failure_rate is not None:
                    self._record_metric(
                        metric_name='validation_failure_rate',
                        metric_type=MetricType.VALIDATION,
                        metric_category=MetricCategory.ERROR_RATE,
                        value=validation_failure_rate,
                        unit='ratio'
                    )
                
                # Get data freshness
                data_freshness = self._get_data_freshness()
                if data_freshness is not None:
                    self._record_metric(
                        metric_name='data_freshness_minutes',
                        metric_type=MetricType.DATA_QUALITY,
                        metric_category=MetricCategory.AVAILABILITY,
                        value=data_freshness,
                        unit='minutes'
                    )
                
                # Wait for next monitoring cycle
                self.stop_monitoring.wait(self.config.get('monitoring_interval_seconds', 60))
                
            except Exception as e:
                logger.error(f"Error in prediction metrics monitoring: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _record_metric(self, metric_name: str, metric_type: MetricType,
                      metric_category: MetricCategory, value: float, unit: str,
                      symbol: str = None, metadata: Dict[str, Any] = None):
        """Record a performance metric"""
        try:
            # Create metric record
            metric = PerformanceMetric(
                metric_name=metric_name,
                metric_type=metric_type,
                metric_category=metric_category,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                symbol=symbol,
                metadata=metadata or {}
            )
            
            # Add to buffer
            if metric_name not in self.metric_buffer:
                self.metric_buffer[metric_name] = deque(maxlen=self.buffer_size)
            
            self.metric_buffer[metric_name].append(metric)
            
            # Store in database
            self.data_storage.store_performance_metric(
                symbol=symbol,
                metric_name=metric_name,
                metric_value=value,
                metric_timestamp=metric.timestamp,
                metadata={
                    'metric_type': metric_type.value,
                    'metric_category': metric_category.value,
                    'unit': unit,
                    **metadata
                }
            )
            
            # Update statistics
            self.performance_stats['total_metrics_collected'] += 1
            self.performance_stats['last_updated'] = datetime.now()
            
            # Check thresholds and create alerts
            self._check_metric_thresholds(metric)
            
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")
    
    def _check_metric_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds and create alerts"""
        try:
            threshold_key = metric.metric_name
            if threshold_key not in self.thresholds:
                return
            
            threshold_value = self.thresholds[threshold_key]
            actual_value = metric.value
            
            # Determine if threshold is exceeded
            threshold_exceeded = False
            alert_type = ""
            
            if 'percent' in metric.unit or 'ratio' in metric.unit:
                # For percentages and ratios, higher is worse
                if actual_value > threshold_value:
                    threshold_exceeded = True
                    alert_type = "HIGH"
            elif 'latency' in metric.metric_name or 'ms' in metric.unit:
                # For latency, higher is worse
                if actual_value > threshold_value:
                    threshold_exceeded = True
                    alert_type = "HIGH"
            elif 'throughput' in metric.metric_name or 'count' in metric.unit:
                # For throughput, lower is worse
                if actual_value < threshold_value:
                    threshold_exceeded = True
                    alert_type = "LOW"
            elif 'accuracy' in metric.metric_name:
                # For accuracy, lower is worse
                if actual_value < threshold_value:
                    threshold_exceeded = True
                    alert_type = "LOW"
            
            if threshold_exceeded:
                self._create_performance_alert(
                    metric_name=metric.metric_name,
                    alert_type=alert_type,
                    threshold_value=threshold_value,
                    actual_value=actual_value,
                    metric=metric
                )
                
                # Call callback if set
                if self.on_metric_threshold_exceeded:
                    self.on_metric_threshold_exceeded(
                        metric.metric_name, actual_value, threshold_value
                    )
            
        except Exception as e:
            logger.error(f"Error checking metric thresholds: {e}")
    
    def _create_performance_alert(self, metric_name: str, alert_type: str,
                                threshold_value: float, actual_value: float,
                                metric: PerformanceMetric):
        """Create a performance alert"""
        try:
            alert_id = f"perf_alert_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Check if similar alert already exists (cooldown)
            existing_alert_key = f"{metric_name}_{alert_type}"
            if existing_alert_key in self.active_alerts:
                existing_alert = self.active_alerts[existing_alert_key]
                cooldown_minutes = self.config.get('alert_cooldown_minutes', 5)
                if (datetime.now() - existing_alert.timestamp).total_seconds() < cooldown_minutes * 60:
                    return  # Skip alert due to cooldown
            
            # Determine severity
            severity = "WARNING"
            if alert_type == "HIGH" and actual_value > threshold_value * 1.5:
                severity = "CRITICAL"
            elif alert_type == "LOW" and actual_value < threshold_value * 0.5:
                severity = "CRITICAL"
            
            # Create alert
            alert = PerformanceAlert(
                alert_id=alert_id,
                metric_name=metric_name,
                alert_type=alert_type,
                severity=severity,
                threshold_value=threshold_value,
                actual_value=actual_value,
                message=f"{metric_name} {alert_type} threshold exceeded: {actual_value:.2f} vs {threshold_value:.2f}",
                timestamp=datetime.now()
            )
            
            # Store alert
            self.active_alerts[existing_alert_key] = alert
            self.alert_history.append(alert)
            
            # Store in database
            self.data_storage.store_system_alert(
                alert_type=f"PERFORMANCE_{alert_type}",
                severity=severity,
                message=alert.message,
                details={
                    'metric_name': metric_name,
                    'threshold_value': threshold_value,
                    'actual_value': actual_value,
                    'unit': metric.unit,
                    'symbol': metric.symbol
                }
            )
            
            # Update statistics
            self.performance_stats['active_alerts'] = len(self.active_alerts)
            
            logger.warning(f"Performance alert created: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error creating performance alert: {e}")
    
    def _get_recent_model_accuracy(self) -> Optional[float]:
        """Get recent model accuracy"""
        try:
            # This would typically query the accuracy tracker or database
            # For now, return a placeholder
            return 0.75  # Placeholder accuracy
            
        except Exception as e:
            logger.error(f"Error getting recent model accuracy: {e}")
            return None
    
    def _get_prediction_count_last_hour(self) -> int:
        """Get prediction count for last hour"""
        try:
            cursor = self.data_storage.connection.cursor()
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            cursor.execute("""
                SELECT COUNT(*) FROM predictions 
                WHERE prediction_timestamp >= ?
            """, (one_hour_ago,))
            
            count = cursor.fetchone()[0]
            return count
            
        except Exception as e:
            logger.error(f"Error getting prediction count: {e}")
            return 0
    
    def _get_average_prediction_latency(self) -> Optional[float]:
        """Get average prediction latency"""
        try:
            # This would typically calculate from prediction timestamps
            # For now, return a placeholder
            return 2500.0  # Placeholder latency in milliseconds
            
        except Exception as e:
            logger.error(f"Error getting prediction latency: {e}")
            return None
    
    def _get_validation_failure_rate(self) -> Optional[float]:
        """Get validation failure rate"""
        try:
            cursor = self.data_storage.connection.cursor()
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            # Get total validations
            cursor.execute("""
                SELECT COUNT(*) FROM validation_results 
                WHERE prediction_timestamp >= ?
            """, (one_hour_ago,))
            total_validations = cursor.fetchone()[0]
            
            if total_validations == 0:
                return None
            
            # Get failed validations
            cursor.execute("""
                SELECT COUNT(*) FROM validation_results 
                WHERE prediction_timestamp >= ? AND overall_status IN ('invalid', 'critical')
            """, (one_hour_ago,))
            failed_validations = cursor.fetchone()[0]
            
            return failed_validations / total_validations
            
        except Exception as e:
            logger.error(f"Error getting validation failure rate: {e}")
            return None
    
    def _get_data_freshness(self) -> Optional[float]:
        """Get data freshness in minutes"""
        try:
            cursor = self.data_storage.connection.cursor()
            
            # Get latest market data timestamp
            cursor.execute("""
                SELECT MAX(timestamp) FROM market_data
            """)
            latest_timestamp = cursor.fetchone()[0]
            
            if not latest_timestamp:
                return None
            
            latest_time = datetime.fromisoformat(latest_timestamp)
            freshness_minutes = (datetime.now() - latest_time).total_seconds() / 60
            
            return freshness_minutes
            
        except Exception as e:
            logger.error(f"Error getting data freshness: {e}")
            return None
    
    def get_performance_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Get metrics from database
            metrics_df = self.data_storage.get_performance_metrics(
                days_back=hours_back // 24 + 1
            )
            
            if metrics_df.empty:
                return {'error': 'No performance data available'}
            
            # Filter by time period
            metrics_df['metric_timestamp'] = pd.to_datetime(metrics_df['metric_timestamp'])
            recent_metrics = metrics_df[metrics_df['metric_timestamp'] >= cutoff_time]
            
            # Calculate summary statistics
            summary = {
                'time_period_hours': hours_back,
                'total_metrics': len(recent_metrics),
                'metrics_by_type': recent_metrics['metric_name'].value_counts().to_dict(),
                'active_alerts': len(self.active_alerts),
                'system_uptime_hours': (datetime.now() - self.performance_stats['system_uptime']).total_seconds() / 3600
            }
            
            # Calculate averages for key metrics
            key_metrics = ['cpu_usage_percent', 'memory_usage_percent', 'prediction_latency_ms', 'model_accuracy']
            for metric in key_metrics:
                metric_data = recent_metrics[recent_metrics['metric_name'] == metric]
                if not metric_data.empty:
                    summary[f'{metric}_avg'] = metric_data['metric_value'].mean()
                    summary[f'{metric}_max'] = metric_data['metric_value'].max()
                    summary[f'{metric}_min'] = metric_data['metric_value'].min()
            
            # Add alert summary
            recent_alerts = [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
            summary['alerts_summary'] = {
                'total_alerts': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.severity == 'CRITICAL']),
                'warning_alerts': len([a for a in recent_alerts if a.severity == 'WARNING']),
                'resolved_alerts': len([a for a in recent_alerts if a.resolved])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def get_metric_trends(self, metric_name: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get trends for a specific metric"""
        try:
            # Get metrics from buffer or database
            if metric_name in self.metric_buffer:
                metrics = list(self.metric_buffer[metric_name])
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            else:
                # Get from database
                metrics_df = self.data_storage.get_performance_metrics(
                    metric_name=metric_name,
                    days_back=hours_back // 24 + 1
                )
                
                if metrics_df.empty:
                    return {'error': f'No data for metric {metric_name}'}
                
                recent_metrics = []
                for _, row in metrics_df.iterrows():
                    metric = PerformanceMetric(
                        metric_name=row['metric_name'],
                        metric_type=MetricType.SYSTEM,  # Default
                        metric_category=MetricCategory.LATENCY,  # Default
                        value=row['metric_value'],
                        unit='unknown',
                        timestamp=row['metric_timestamp']
                    )
                    recent_metrics.append(metric)
            
            if not recent_metrics:
                return {'error': f'No recent data for metric {metric_name}'}
            
            # Calculate trends
            values = [m.value for m in recent_metrics]
            timestamps = [m.timestamp for m in recent_metrics]
            
            # Calculate trend direction
            if len(values) >= 2:
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                if trend_slope > 0.01:
                    trend_direction = "increasing"
                elif trend_slope < -0.01:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "insufficient_data"
            
            return {
                'metric_name': metric_name,
                'time_period_hours': hours_back,
                'data_points': len(recent_metrics),
                'current_value': values[-1] if values else None,
                'average_value': np.mean(values),
                'min_value': np.min(values),
                'max_value': np.max(values),
                'std_value': np.std(values),
                'trend_direction': trend_direction,
                'values': values,
                'timestamps': [t.isoformat() for t in timestamps]
            }
            
        except Exception as e:
            logger.error(f"Error getting metric trends for {metric_name}: {e}")
            return {'error': str(e)}
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a performance alert"""
        try:
            # Find alert in active alerts
            for key, alert in self.active_alerts.items():
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    del self.active_alerts[key]
                    
                    # Update statistics
                    self.performance_stats['active_alerts'] = len(self.active_alerts)
                    
                    logger.info(f"Resolved performance alert: {alert_id}")
                    return True
            
            # Find alert in history
            for alert in self.alert_history:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    logger.info(f"Resolved performance alert: {alert_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop all monitoring threads"""
        try:
            self.stop_monitoring.set()
            
            # Wait for threads to finish
            for thread_name, thread in self.monitoring_threads.items():
                if thread.is_alive():
                    thread.join(timeout=10)
            
            logger.info("Performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        try:
            return {
                'monitoring_active': not self.stop_monitoring.is_set(),
                'monitoring_threads': {
                    name: thread.is_alive() for name, thread in self.monitoring_threads.items()
                },
                'performance_stats': self.performance_stats,
                'active_alerts': len(self.active_alerts),
                'metric_buffers': {
                    name: len(buffer) for name, buffer in self.metric_buffer.items()
                },
                'thresholds': self.thresholds,
                'config': self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("PerformanceMonitor module loaded successfully")
    print("Use with DataStorage instance for full functionality")