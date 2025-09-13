"""
Performance Optimization System for Real-time LSTM Prediction System
Phase 2.6: Performance optimizations, caching, and resource management
"""

import numpy as np
import pandas as pd
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, wraps
import hashlib
import pickle
import os
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class CacheType(Enum):
    """Cache type enumeration"""
    PREDICTION = "prediction"
    MODEL = "model"
    DATA = "data"
    CONFIGURATION = "configuration"
    METRICS = "metrics"

@dataclass
class CacheEntry:
    """Cache entry data structure"""
    key: str
    value: Any
    timestamp: datetime
    expiry_time: datetime
    access_count: int = 0
    last_access: datetime = None
    size_bytes: int = 0

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cache_hit_rate: float
    timestamp: datetime

class OptimizationSystem:
    """
    Performance optimization system with caching, resource management, and monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the optimization system
        
        Args:
            config: Configuration dictionary with optimization parameters
        """
        self.config = config
        
        # Cache configuration
        self.cache_config = config.get('cache', {})
        self.max_cache_size_mb = self.cache_config.get('max_size_mb', 500)
        self.cache_ttl_minutes = self.cache_config.get('ttl_minutes', 60)
        self.enable_prediction_cache = self.cache_config.get('enable_prediction_cache', True)
        self.enable_model_cache = self.cache_config.get('enable_model_cache', True)
        self.enable_data_cache = self.cache_config.get('enable_data_cache', True)
        
        # Performance monitoring
        self.performance_config = config.get('performance', {})
        self.enable_performance_monitoring = self.performance_config.get('enable_monitoring', True)
        self.performance_history_size = self.performance_config.get('history_size', 1000)
        
        # Resource management
        self.resource_config = config.get('resources', {})
        self.max_memory_usage_mb = self.resource_config.get('max_memory_mb', 2000)
        self.gc_interval_minutes = self.resource_config.get('gc_interval_minutes', 30)
        
        # Initialize caches
        self.caches = {
            CacheType.PREDICTION: {},
            CacheType.MODEL: {},
            CacheType.DATA: {},
            CacheType.CONFIGURATION: {},
            CacheType.METRICS: {}
        }
        
        # Performance tracking
        self.performance_metrics = deque(maxlen=self.performance_history_size)
        self.operation_times = defaultdict(list)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_bytes': 0
        }
        
        # Resource monitoring
        self.memory_usage_history = deque(maxlen=100)
        self.last_gc_time = datetime.now()
        
        # Background tasks
        self.background_tasks_active = False
        self.background_thread = None
        
        logger.info("OptimizationSystem initialized")
    
    def start_optimization(self):
        """Start optimization background tasks"""
        if self.background_tasks_active:
            logger.warning("Optimization tasks already active")
            return
        
        self.background_tasks_active = True
        self.background_thread = threading.Thread(target=self._background_optimization_loop, daemon=True)
        self.background_thread.start()
        
        logger.info("Optimization system started")
    
    def stop_optimization(self):
        """Stop optimization background tasks"""
        self.background_tasks_active = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        
        logger.info("Optimization system stopped")
    
    def cache_prediction(self, symbol: str, prediction_data: Dict[str, Any], 
                        ttl_minutes: Optional[int] = None) -> str:
        """
        Cache prediction data
        
        Args:
            symbol: Stock symbol
            prediction_data: Prediction data to cache
            ttl_minutes: Time to live in minutes
            
        Returns:
            Cache key
        """
        try:
            if not self.enable_prediction_cache:
                return None
            
            cache_key = self._generate_cache_key('prediction', symbol, prediction_data)
            ttl = ttl_minutes or self.cache_ttl_minutes
            
            self._store_cache_entry(CacheType.PREDICTION, cache_key, prediction_data, ttl)
            
            logger.debug(f"Cached prediction for {symbol}")
            return cache_key
            
        except Exception as e:
            logger.error(f"Error caching prediction for {symbol}: {e}")
            return None
    
    def get_cached_prediction(self, symbol: str, prediction_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction data
        
        Args:
            symbol: Stock symbol
            prediction_params: Prediction parameters
            
        Returns:
            Cached prediction data or None
        """
        try:
            if not self.enable_prediction_cache:
                return None
            
            cache_key = self._generate_cache_key('prediction', symbol, prediction_params)
            return self._get_cache_entry(CacheType.PREDICTION, cache_key)
            
        except Exception as e:
            logger.error(f"Error getting cached prediction for {symbol}: {e}")
            return None
    
    def cache_model_data(self, symbol: str, model_data: Any, ttl_minutes: Optional[int] = None) -> str:
        """
        Cache model data
        
        Args:
            symbol: Stock symbol
            model_data: Model data to cache
            ttl_minutes: Time to live in minutes
            
        Returns:
            Cache key
        """
        try:
            if not self.enable_model_cache:
                return None
            
            cache_key = f"model_{symbol}_{int(time.time())}"
            ttl = ttl_minutes or (self.cache_ttl_minutes * 24)  # Models cache longer
            
            self._store_cache_entry(CacheType.MODEL, cache_key, model_data, ttl)
            
            logger.debug(f"Cached model data for {symbol}")
            return cache_key
            
        except Exception as e:
            logger.error(f"Error caching model data for {symbol}: {e}")
            return None
    
    def get_cached_model_data(self, symbol: str) -> Optional[Any]:
        """
        Get cached model data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Cached model data or None
        """
        try:
            if not self.enable_model_cache:
                return None
            
            # Find the most recent model cache for this symbol
            for cache_key in self.caches[CacheType.MODEL]:
                if cache_key.startswith(f"model_{symbol}_"):
                    return self._get_cache_entry(CacheType.MODEL, cache_key)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached model data for {symbol}: {e}")
            return None
    
    def cache_market_data(self, symbol: str, data: pd.DataFrame, 
                         ttl_minutes: Optional[int] = None) -> str:
        """
        Cache market data
        
        Args:
            symbol: Stock symbol
            data: Market data DataFrame
            ttl_minutes: Time to live in minutes
            
        Returns:
            Cache key
        """
        try:
            if not self.enable_data_cache:
                return None
            
            cache_key = f"data_{symbol}_{data.index[-1].strftime('%Y%m%d_%H%M')}"
            ttl = ttl_minutes or (self.cache_ttl_minutes // 2)  # Data caches shorter
            
            self._store_cache_entry(CacheType.DATA, cache_key, data, ttl)
            
            logger.debug(f"Cached market data for {symbol}")
            return cache_key
            
        except Exception as e:
            logger.error(f"Error caching market data for {symbol}: {e}")
            return None
    
    def get_cached_market_data(self, symbol: str, start_time: datetime, 
                              end_time: datetime) -> Optional[pd.DataFrame]:
        """
        Get cached market data
        
        Args:
            symbol: Stock symbol
            start_time: Start time
            end_time: End time
            
        Returns:
            Cached market data or None
        """
        try:
            if not self.enable_data_cache:
                return None
            
            # Find cached data that covers the requested time range
            for cache_key in self.caches[CacheType.DATA]:
                if cache_key.startswith(f"data_{symbol}_"):
                    cached_data = self._get_cache_entry(CacheType.DATA, cache_key)
                    if cached_data is not None:
                        # Check if cached data covers the requested range
                        if (cached_data.index[0] <= start_time and 
                            cached_data.index[-1] >= end_time):
                            return cached_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached market data for {symbol}: {e}")
            return None
    
    def performance_monitor(self, operation_name: str):
        """
        Decorator for performance monitoring
        
        Args:
            operation_name: Name of the operation to monitor
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_performance_monitoring:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    duration_ms = (end_time - start_time) * 1000
                    memory_usage_mb = end_memory - start_memory
                    
                    self._record_performance_metric(
                        operation_name, duration_ms, memory_usage_mb
                    )
            
            return wrapper
        return decorator
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for memory usage
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        try:
            original_size = df.memory_usage(deep=True).sum()
            
            # Optimize data types
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    try:
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    except:
                        try:
                            df[col] = pd.to_numeric(df[col], downcast='float')
                        except:
                            pass
                elif df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            
            # Remove unnecessary columns
            df = df.dropna(axis=1, how='all')
            
            optimized_size = df.memory_usage(deep=True).sum()
            reduction_percent = (original_size - optimized_size) / original_size * 100
            
            logger.debug(f"DataFrame optimized: {reduction_percent:.1f}% size reduction")
            
            return df
            
        except Exception as e:
            logger.error(f"Error optimizing DataFrame: {e}")
            return df
    
    def batch_operations(self, operations: List[Callable], batch_size: int = 10) -> List[Any]:
        """
        Execute operations in batches for better performance
        
        Args:
            operations: List of operations to execute
            batch_size: Size of each batch
            
        Returns:
            List of results
        """
        try:
            results = []
            
            for i in range(0, len(operations), batch_size):
                batch = operations[i:i + batch_size]
                
                # Execute batch
                batch_results = []
                for operation in batch:
                    try:
                        result = operation()
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in batch operation: {e}")
                        batch_results.append(None)
                
                results.extend(batch_results)
                
                # Small delay between batches to prevent resource exhaustion
                if i + batch_size < len(operations):
                    time.sleep(0.01)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch operations: {e}")
            return []
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance optimization report
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Calculate cache statistics
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            cache_hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate average operation times
            avg_operation_times = {}
            for operation, times in self.operation_times.items():
                if times:
                    avg_operation_times[operation] = np.mean(times)
            
            # Calculate memory usage
            current_memory = self._get_memory_usage()
            memory_history = list(self.memory_usage_history)
            avg_memory = np.mean(memory_history) if memory_history else 0
            
            # Calculate cache size
            total_cache_size = self.cache_stats['total_size_bytes'] / (1024 * 1024)  # MB
            
            return {
                'cache_statistics': {
                    'hit_rate_percent': cache_hit_rate,
                    'total_hits': self.cache_stats['hits'],
                    'total_misses': self.cache_stats['misses'],
                    'total_evictions': self.cache_stats['evictions'],
                    'total_size_mb': total_cache_size,
                    'max_size_mb': self.max_cache_size_mb
                },
                'performance_metrics': {
                    'total_operations': len(self.performance_metrics),
                    'average_operation_times': avg_operation_times,
                    'current_memory_usage_mb': current_memory,
                    'average_memory_usage_mb': avg_memory,
                    'max_memory_limit_mb': self.max_memory_usage_mb
                },
                'optimization_status': {
                    'prediction_cache_enabled': self.enable_prediction_cache,
                    'model_cache_enabled': self.enable_model_cache,
                    'data_cache_enabled': self.enable_data_cache,
                    'performance_monitoring_enabled': self.enable_performance_monitoring,
                    'background_tasks_active': self.background_tasks_active
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _background_optimization_loop(self):
        """Background optimization loop"""
        while self.background_tasks_active:
            try:
                # Clean expired cache entries
                self._clean_expired_cache_entries()
                
                # Check memory usage and cleanup if needed
                self._check_memory_usage()
                
                # Update memory usage history
                self._update_memory_history()
                
                # Sleep for 5 minutes before next optimization cycle
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in background optimization loop: {e}")
                time.sleep(60)  # Sleep for 1 minute on error
    
    def _generate_cache_key(self, cache_type: str, symbol: str, data: Any) -> str:
        """Generate cache key"""
        try:
            # Create a hash of the data for uniqueness
            data_str = str(data) if not isinstance(data, (dict, list)) else str(sorted(data.items()) if isinstance(data, dict) else data)
            data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
            
            return f"{cache_type}_{symbol}_{data_hash}_{int(time.time())}"
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"{cache_type}_{symbol}_{int(time.time())}"
    
    def _store_cache_entry(self, cache_type: CacheType, key: str, value: Any, ttl_minutes: int):
        """Store cache entry"""
        try:
            # Calculate size
            size_bytes = len(pickle.dumps(value))
            
            # Check if we need to evict entries
            self._check_cache_size(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                expiry_time=datetime.now() + timedelta(minutes=ttl_minutes),
                size_bytes=size_bytes
            )
            
            # Store in cache
            self.caches[cache_type][key] = entry
            self.cache_stats['total_size_bytes'] += size_bytes
            
            logger.debug(f"Stored cache entry: {key} ({size_bytes} bytes)")
            
        except Exception as e:
            logger.error(f"Error storing cache entry: {e}")
    
    def _get_cache_entry(self, cache_type: CacheType, key: str) -> Optional[Any]:
        """Get cache entry"""
        try:
            if key not in self.caches[cache_type]:
                self.cache_stats['misses'] += 1
                return None
            
            entry = self.caches[cache_type][key]
            
            # Check if expired
            if datetime.now() > entry.expiry_time:
                del self.caches[cache_type][key]
                self.cache_stats['total_size_bytes'] -= entry.size_bytes
                self.cache_stats['misses'] += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = datetime.now()
            self.cache_stats['hits'] += 1
            
            return entry.value
            
        except Exception as e:
            logger.error(f"Error getting cache entry: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def _clean_expired_cache_entries(self):
        """Clean expired cache entries"""
        try:
            current_time = datetime.now()
            total_cleaned = 0
            
            for cache_type in self.caches:
                expired_keys = []
                for key, entry in self.caches[cache_type].items():
                    if current_time > entry.expiry_time:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    entry = self.caches[cache_type][key]
                    del self.caches[cache_type][key]
                    self.cache_stats['total_size_bytes'] -= entry.size_bytes
                    total_cleaned += 1
            
            if total_cleaned > 0:
                logger.debug(f"Cleaned {total_cleaned} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning expired cache entries: {e}")
    
    def _check_cache_size(self, new_entry_size: int):
        """Check cache size and evict if necessary"""
        try:
            max_size_bytes = self.max_cache_size_mb * 1024 * 1024
            
            if self.cache_stats['total_size_bytes'] + new_entry_size > max_size_bytes:
                # Evict least recently used entries
                self._evict_lru_entries(new_entry_size)
                
        except Exception as e:
            logger.error(f"Error checking cache size: {e}")
    
    def _evict_lru_entries(self, required_space: int):
        """Evict least recently used cache entries"""
        try:
            # Collect all entries with access times
            all_entries = []
            for cache_type in self.caches:
                for key, entry in self.caches[cache_type].items():
                    all_entries.append((cache_type, key, entry))
            
            # Sort by last access time (oldest first)
            all_entries.sort(key=lambda x: x[2].last_access or x[2].timestamp)
            
            # Evict entries until we have enough space
            freed_space = 0
            evicted_count = 0
            
            for cache_type, key, entry in all_entries:
                if freed_space >= required_space:
                    break
                
                del self.caches[cache_type][key]
                self.cache_stats['total_size_bytes'] -= entry.size_bytes
                freed_space += entry.size_bytes
                evicted_count += 1
            
            self.cache_stats['evictions'] += evicted_count
            
            if evicted_count > 0:
                logger.debug(f"Evicted {evicted_count} cache entries, freed {freed_space} bytes")
                
        except Exception as e:
            logger.error(f"Error evicting LRU entries: {e}")
    
    def _check_memory_usage(self):
        """Check memory usage and cleanup if needed"""
        try:
            current_memory = self._get_memory_usage()
            
            if current_memory > self.max_memory_usage_mb:
                logger.warning(f"High memory usage: {current_memory:.1f}MB")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear some caches if still high
                new_memory = self._get_memory_usage()
                if new_memory > self.max_memory_usage_mb * 0.9:
                    self._clear_old_cache_entries()
                
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
    
    def _clear_old_cache_entries(self):
        """Clear old cache entries to free memory"""
        try:
            # Clear data cache (usually largest)
            if CacheType.DATA in self.caches:
                old_size = sum(entry.size_bytes for entry in self.caches[CacheType.DATA].values())
                self.caches[CacheType.DATA].clear()
                self.cache_stats['total_size_bytes'] -= old_size
                logger.info(f"Cleared data cache, freed {old_size} bytes")
                
        except Exception as e:
            logger.error(f"Error clearing old cache entries: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    def _update_memory_history(self):
        """Update memory usage history"""
        try:
            current_memory = self._get_memory_usage()
            self.memory_usage_history.append(current_memory)
        except Exception as e:
            logger.error(f"Error updating memory history: {e}")
    
    def _record_performance_metric(self, operation: str, duration_ms: float, memory_usage_mb: float):
        """Record performance metric"""
        try:
            metric = PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage_mb,
                cache_hit_rate=0.0,  # Will be calculated separately
                timestamp=datetime.now()
            )
            
            self.performance_metrics.append(metric)
            self.operation_times[operation].append(duration_ms)
            
            # Keep only recent operation times
            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation] = self.operation_times[operation][-100:]
                
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = {
        'cache': {
            'max_size_mb': 100,
            'ttl_minutes': 30,
            'enable_prediction_cache': True,
            'enable_model_cache': True,
            'enable_data_cache': True
        },
        'performance': {
            'enable_monitoring': True,
            'history_size': 1000
        },
        'resources': {
            'max_memory_mb': 1000,
            'gc_interval_minutes': 30
        }
    }
    
    # Test the optimization system
    optimizer = OptimizationSystem(config)
    
    # Test caching
    test_data = {'prediction': 100.0, 'confidence': 0.8}
    cache_key = optimizer.cache_prediction('AAPL', test_data)
    print(f"Cached prediction with key: {cache_key}")
    
    # Test retrieval
    retrieved_data = optimizer.get_cached_prediction('AAPL', test_data)
    print(f"Retrieved data: {retrieved_data}")
    
    # Test performance monitoring
    @optimizer.performance_monitor('test_operation')
    def test_operation():
        time.sleep(0.1)
        return "test_result"
    
    result = test_operation()
    print(f"Test operation result: {result}")
    
    # Test performance report
    report = optimizer.get_performance_report()
    print(f"Performance report: {report}")
    
    print("Optimization system test completed")
