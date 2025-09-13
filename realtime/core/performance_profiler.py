"""
Performance Profiling System for Real-time LSTM Prediction System
Phase 4.3: Comprehensive performance profiling and optimization
"""

import time
import threading
import psutil
import cProfile
import pstats
import io
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import queue
import uuid
from contextlib import contextmanager
import functools
import tracemalloc
import gc

logger = logging.getLogger(__name__)

class ProfileType(Enum):
    """Profile types"""
    FUNCTION = "function"
    BLOCK = "block"
    SYSTEM = "system"
    MEMORY = "memory"
    CPU = "cpu"
    I_O = "io"

class ProfileMetric(Enum):
    """Profile metrics"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    I_O_OPERATIONS = "io_operations"
    FUNCTION_CALLS = "function_calls"
    CACHE_HITS = "cache_hits"
    CACHE_MISSES = "cache_misses"

@dataclass
class ProfileRecord:
    """Profile record"""
    profile_id: str
    timestamp: datetime
    profile_type: ProfileType
    metric_type: ProfileMetric
    operation: str
    value: float
    unit: str
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FunctionProfile:
    """Function profiling data"""
    function_name: str
    total_calls: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    cumulative_time: float
    last_called: datetime
    call_frequency: float  # calls per second

@dataclass
class SystemProfile:
    """System profiling data"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]

class MemoryProfiler:
    """Memory profiling utilities"""
    
    def __init__(self):
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.memory_traces: List[Dict[str, Any]] = []
        
    def start_tracing(self):
        """Start memory tracing"""
        tracemalloc.start()
        
    def stop_tracing(self):
        """Stop memory tracing"""
        tracemalloc.stop()
    
    def take_snapshot(self) -> Dict[str, Any]:
        """Take memory snapshot"""
        try:
            # Get process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            snapshot = {
                'timestamp': datetime.now(),
                'process_memory': {
                    'rss': memory_info.rss,  # Resident Set Size
                    'vms': memory_info.vms,  # Virtual Memory Size
                    'percent': process.memory_percent()
                },
                'system_memory': {
                    'total': system_memory.total,
                    'available': system_memory.available,
                    'percent': system_memory.percent,
                    'used': system_memory.used,
                    'free': system_memory.free
                },
                'gc_stats': {
                    'counts': gc.get_count(),
                    'threshold': gc.get_threshold()
                }
            }
            
            # Get tracemalloc snapshot if available
            if tracemalloc.is_tracing():
                snapshot['tracemalloc'] = self._get_tracemalloc_snapshot()
            
            self.memory_snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            logger.error(f"Error taking memory snapshot: {e}")
            return {}
    
    def _get_tracemalloc_snapshot(self) -> Dict[str, Any]:
        """Get tracemalloc snapshot"""
        try:
            snapshot = tracemalloc.take_snapshot()
            
            # Get top memory allocations
            top_stats = snapshot.statistics('lineno')
            
            return {
                'total_size': sum(stat.size for stat in top_stats),
                'total_count': sum(stat.count for stat in top_stats),
                'top_allocations': [
                    {
                        'filename': stat.traceback.format()[0],
                        'size': stat.size,
                        'count': stat.count
                    }
                    for stat in top_stats[:10]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting tracemalloc snapshot: {e}")
            return {}
    
    def compare_snapshots(self, snapshot1: Dict[str, Any], snapshot2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two memory snapshots"""
        try:
            comparison = {
                'timestamp1': snapshot1['timestamp'],
                'timestamp2': snapshot2['timestamp'],
                'process_memory_diff': {
                    'rss_diff': snapshot2['process_memory']['rss'] - snapshot1['process_memory']['rss'],
                    'vms_diff': snapshot2['process_memory']['vms'] - snapshot1['process_memory']['vms'],
                    'percent_diff': snapshot2['process_memory']['percent'] - snapshot1['process_memory']['percent']
                },
                'system_memory_diff': {
                    'available_diff': snapshot2['system_memory']['available'] - snapshot1['system_memory']['available'],
                    'percent_diff': snapshot2['system_memory']['percent'] - snapshot1['system_memory']['percent']
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing memory snapshots: {e}")
            return {}

class CPUProfiler:
    """CPU profiling utilities"""
    
    def __init__(self):
        self.cpu_samples: List[Dict[str, Any]] = []
        
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function's CPU usage"""
        try:
            # Start profiling
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            
            # Execute function
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_cpu = psutil.cpu_percent()
            
            # Stop profiling
            profiler.disable()
            
            # Get profiling stats
            stats_buffer = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_buffer)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            profile_data = {
                'function_name': func.__name__,
                'execution_time': end_time - start_time,
                'cpu_usage_start': start_cpu,
                'cpu_usage_end': end_cpu,
                'cpu_usage_avg': (start_cpu + end_cpu) / 2,
                'profile_stats': stats_buffer.getvalue(),
                'result': result
            }
            
            self.cpu_samples.append(profile_data)
            return profile_data
            
        except Exception as e:
            logger.error(f"Error profiling function: {e}")
            return {}
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        return psutil.cpu_percent(interval=1)
    
    def get_cpu_count(self) -> int:
        """Get CPU count"""
        return psutil.cpu_count()

class I_OProfiler:
    """I/O profiling utilities"""
    
    def __init__(self):
        self.io_operations: List[Dict[str, Any]] = []
        
    def profile_io_operation(self, operation: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile an I/O operation"""
        try:
            # Get initial I/O stats
            initial_io = psutil.Process().io_counters()
            
            start_time = time.time()
            
            # Execute I/O operation
            result = func(*args, **kwargs)
            
            end_time = time.time()
            
            # Get final I/O stats
            final_io = psutil.Process().io_counters()
            
            io_data = {
                'operation': operation,
                'execution_time': end_time - start_time,
                'read_bytes': final_io.read_bytes - initial_io.read_bytes,
                'write_bytes': final_io.write_bytes - initial_io.write_bytes,
                'read_count': final_io.read_count - initial_io.read_count,
                'write_count': final_io.write_count - initial_io.write_count,
                'result': result
            }
            
            self.io_operations.append(io_data)
            return io_data
            
        except Exception as e:
            logger.error(f"Error profiling I/O operation: {e}")
            return {}

class PerformanceProfiler:
    """Main performance profiling system"""
    
    def __init__(self, data_storage, config: Dict[str, Any]):
        self.data_storage = data_storage
        self.config = config
        
        # Profiling components
        self.memory_profiler = MemoryProfiler()
        self.cpu_profiler = CPUProfiler()
        self.io_profiler = I_OProfiler()
        
        # Profile storage
        self.profiles: Dict[str, ProfileRecord] = {}
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.system_profiles: List[SystemProfile] = []
        
        # Profiling queue
        self.profile_queue = queue.Queue(maxsize=config.get('profile_queue_size', 1000))
        
        # Processing
        self.running = False
        self.processor_thread = None
        
        # Statistics
        self.stats = {
            'total_profiles': 0,
            'profiles_by_type': defaultdict(int),
            'profiles_by_metric': defaultdict(int),
            'last_profile': None
        }
        
        # Performance thresholds
        self.thresholds = config.get('performance_thresholds', {
            'max_execution_time_ms': 5000,
            'max_memory_usage_mb': 1000,
            'max_cpu_usage_percent': 80
        })
        
        # Start processing
        self.start()
        
        logger.info("PerformanceProfiler initialized")
    
    def start(self):
        """Start performance profiling"""
        if self.running:
            return
        
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_profiles, daemon=True)
        self.processor_thread.start()
        
        # Start system monitoring if enabled
        if self.config.get('enable_system_monitoring', True):
            self._start_system_monitoring()
        
        logger.info("PerformanceProfiler started")
    
    def stop(self):
        """Stop performance profiling"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        
        # Stop memory tracing
        self.memory_profiler.stop_tracing()
        
        logger.info("PerformanceProfiler stopped")
    
    def _process_profiles(self):
        """Process profiles from queue"""
        while self.running:
            try:
                # Get profile from queue
                profile = self.profile_queue.get(timeout=1)
                
                # Store profile
                self.profiles[profile.profile_id] = profile
                self._store_profile(profile)
                
                # Update statistics
                self._update_stats(profile)
                
                # Check thresholds
                self._check_thresholds(profile)
                
                self.profile_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing profile: {e}")
    
    def _store_profile(self, profile: ProfileRecord):
        """Store profile in database"""
        try:
            self.data_storage.store_performance_profile(
                profile_id=profile.profile_id,
                timestamp=profile.timestamp,
                profile_type=profile.profile_type.value,
                metric_type=profile.metric_type.value,
                operation=profile.operation,
                value=profile.value,
                unit=profile.unit,
                context=json.dumps(profile.context),
                metadata=json.dumps(profile.metadata)
            )
        except Exception as e:
            logger.error(f"Error storing profile: {e}")
    
    def _update_stats(self, profile: ProfileRecord):
        """Update profiling statistics"""
        self.stats['total_profiles'] += 1
        self.stats['profiles_by_type'][profile.profile_type.value] += 1
        self.stats['profiles_by_metric'][profile.metric_type.value] += 1
        self.stats['last_profile'] = profile.timestamp
    
    def _check_thresholds(self, profile: ProfileRecord):
        """Check performance thresholds"""
        try:
            if profile.metric_type == ProfileMetric.EXECUTION_TIME:
                threshold = self.thresholds.get('max_execution_time_ms', 5000)
                if profile.value > threshold:
                    self._create_performance_alert(profile, 'slow_execution')
            
            elif profile.metric_type == ProfileMetric.MEMORY_USAGE:
                threshold = self.thresholds.get('max_memory_usage_mb', 1000) * 1024 * 1024  # Convert to bytes
                if profile.value > threshold:
                    self._create_performance_alert(profile, 'high_memory_usage')
            
            elif profile.metric_type == ProfileMetric.CPU_USAGE:
                threshold = self.thresholds.get('max_cpu_usage_percent', 80)
                if profile.value > threshold:
                    self._create_performance_alert(profile, 'high_cpu_usage')
                    
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
    
    def _create_performance_alert(self, profile: ProfileRecord, alert_type: str):
        """Create performance alert"""
        try:
            self.data_storage.store_system_alert(
                alert_type=f'PERFORMANCE_{alert_type.upper()}',
                severity='WARNING',
                message=f'Performance threshold exceeded: {profile.operation} - {profile.value:.2f} {profile.unit}',
                details={
                    'profile_id': profile.profile_id,
                    'operation': profile.operation,
                    'value': profile.value,
                    'unit': profile.unit,
                    'threshold_type': alert_type
                }
            )
        except Exception as e:
            logger.error(f"Error creating performance alert: {e}")
    
    def _start_system_monitoring(self):
        """Start system monitoring thread"""
        def monitor_system():
            while self.running:
                try:
                    # Get system metrics
                    system_profile = SystemProfile(
                        timestamp=datetime.now(),
                        cpu_percent=psutil.cpu_percent(),
                        memory_percent=psutil.virtual_memory().percent,
                        memory_available=psutil.virtual_memory().available,
                        disk_usage_percent=psutil.disk_usage('/').percent,
                        network_io={
                            'bytes_sent': psutil.net_io_counters().bytes_sent,
                            'bytes_recv': psutil.net_io_counters().bytes_recv
                        },
                        process_count=len(psutil.pids()),
                        load_average=psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                    )
                    
                    self.system_profiles.append(system_profile)
                    
                    # Keep only recent profiles
                    if len(self.system_profiles) > 1000:
                        self.system_profiles = self.system_profiles[-500:]
                    
                    time.sleep(10)  # Monitor every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    time.sleep(60)  # Wait before retrying
        
        thread = threading.Thread(target=monitor_system, daemon=True)
        thread.start()
    
    def profile_function(self, func: Callable, operation: str = None, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function"""
        try:
            operation = operation or func.__name__
            
            # Profile CPU usage
            cpu_profile = self.cpu_profiler.profile_function(func, *args, **kwargs)
            
            # Create profile record
            profile_id = str(uuid.uuid4())
            profile = ProfileRecord(
                profile_id=profile_id,
                timestamp=datetime.now(),
                profile_type=ProfileType.FUNCTION,
                metric_type=ProfileMetric.EXECUTION_TIME,
                operation=operation,
                value=cpu_profile.get('execution_time', 0) * 1000,  # Convert to milliseconds
                unit='milliseconds',
                context={
                    'function_name': func.__name__,
                    'module': func.__module__ if hasattr(func, '__module__') else 'unknown'
                },
                metadata=cpu_profile
            )
            
            # Queue for processing
            self.profile_queue.put(profile)
            
            # Update function profile
            self._update_function_profile(func.__name__, cpu_profile.get('execution_time', 0))
            
            return cpu_profile
            
        except Exception as e:
            logger.error(f"Error profiling function: {e}")
            return {}
    
    def _update_function_profile(self, function_name: str, execution_time: float):
        """Update function profile statistics"""
        try:
            if function_name not in self.function_profiles:
                self.function_profiles[function_name] = FunctionProfile(
                    function_name=function_name,
                    total_calls=0,
                    total_time=0.0,
                    average_time=0.0,
                    min_time=float('inf'),
                    max_time=0.0,
                    cumulative_time=0.0,
                    last_called=datetime.now(),
                    call_frequency=0.0
                )
            
            profile = self.function_profiles[function_name]
            profile.total_calls += 1
            profile.total_time += execution_time
            profile.average_time = profile.total_time / profile.total_calls
            profile.min_time = min(profile.min_time, execution_time)
            profile.max_time = max(profile.max_time, execution_time)
            profile.last_called = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating function profile: {e}")
    
    def profile_memory_operation(self, operation: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a memory operation"""
        try:
            # Take initial memory snapshot
            initial_snapshot = self.memory_profiler.take_snapshot()
            
            start_time = time.time()
            
            # Execute operation
            result = func(*args, **kwargs)
            
            end_time = time.time()
            
            # Take final memory snapshot
            final_snapshot = self.memory_profiler.take_snapshot()
            
            # Compare snapshots
            memory_diff = self.memory_profiler.compare_snapshots(initial_snapshot, final_snapshot)
            
            # Create profile record
            profile_id = str(uuid.uuid4())
            profile = ProfileRecord(
                profile_id=profile_id,
                timestamp=datetime.now(),
                profile_type=ProfileType.MEMORY,
                metric_type=ProfileMetric.MEMORY_USAGE,
                operation=operation,
                value=memory_diff.get('process_memory_diff', {}).get('rss_diff', 0),
                unit='bytes',
                context={'operation': operation},
                metadata={
                    'initial_snapshot': initial_snapshot,
                    'final_snapshot': final_snapshot,
                    'memory_diff': memory_diff,
                    'execution_time': end_time - start_time
                }
            )
            
            # Queue for processing
            self.profile_queue.put(profile)
            
            return {
                'execution_time': end_time - start_time,
                'memory_diff': memory_diff,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error profiling memory operation: {e}")
            return {}
    
    def profile_io_operation(self, operation: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile an I/O operation"""
        try:
            io_profile = self.io_profiler.profile_io_operation(operation, func, *args, **kwargs)
            
            # Create profile record
            profile_id = str(uuid.uuid4())
            profile = ProfileRecord(
                profile_id=profile_id,
                timestamp=datetime.now(),
                profile_type=ProfileType.I_O,
                metric_type=ProfileMetric.I_O_OPERATIONS,
                operation=operation,
                value=io_profile.get('read_bytes', 0) + io_profile.get('write_bytes', 0),
                unit='bytes',
                context={'operation': operation},
                metadata=io_profile
            )
            
            # Queue for processing
            self.profile_queue.put(profile)
            
            return io_profile
            
        except Exception as e:
            logger.error(f"Error profiling I/O operation: {e}")
            return {}
    
    def get_performance_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_profiles = [p for p in self.profiles.values() if p.timestamp >= cutoff_time]
            
            summary = {
                'time_period_hours': hours_back,
                'total_profiles': len(recent_profiles),
                'profiles_by_type': defaultdict(int),
                'profiles_by_metric': defaultdict(int),
                'function_profiles': {},
                'system_profile_summary': self._get_system_profile_summary(hours_back),
                'performance_alerts': self._get_recent_performance_alerts(hours_back)
            }
            
            # Count profiles by type and metric
            for profile in recent_profiles:
                summary['profiles_by_type'][profile.profile_type.value] += 1
                summary['profiles_by_metric'][profile.metric_type.value] += 1
            
            # Get function profiles
            for func_name, func_profile in self.function_profiles.items():
                if func_profile.last_called >= cutoff_time:
                    summary['function_profiles'][func_name] = asdict(func_profile)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def _get_system_profile_summary(self, hours_back: int) -> Dict[str, Any]:
        """Get system profile summary"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_profiles = [p for p in self.system_profiles if p.timestamp >= cutoff_time]
            
            if not recent_profiles:
                return {}
            
            # Calculate averages
            cpu_avg = sum(p.cpu_percent for p in recent_profiles) / len(recent_profiles)
            memory_avg = sum(p.memory_percent for p in recent_profiles) / len(recent_profiles)
            disk_avg = sum(p.disk_usage_percent for p in recent_profiles) / len(recent_profiles)
            
            return {
                'cpu_percent_avg': cpu_avg,
                'memory_percent_avg': memory_avg,
                'disk_usage_percent_avg': disk_avg,
                'profile_count': len(recent_profiles),
                'last_profile': recent_profiles[-1].timestamp if recent_profiles else None
            }
            
        except Exception as e:
            logger.error(f"Error getting system profile summary: {e}")
            return {}
    
    def _get_recent_performance_alerts(self, hours_back: int) -> List[Dict[str, Any]]:
        """Get recent performance alerts"""
        try:
            # This would typically query the database for performance alerts
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent performance alerts: {e}")
            return []
    
    def get_profile_statistics(self) -> Dict[str, Any]:
        """Get profiling statistics"""
        return {
            'stats': self.stats,
            'function_profiles': len(self.function_profiles),
            'system_profiles': len(self.system_profiles),
            'queue_size': self.profile_queue.qsize(),
            'running': self.running,
            'thresholds': self.thresholds
        }

# Context managers for profiling
@contextmanager
def profile_function(profiler: PerformanceProfiler, operation: str = None):
    """Context manager for profiling a function"""
    start_time = time.time()
    operation = operation or 'unknown_operation'
    
    try:
        yield
    finally:
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        profile_id = str(uuid.uuid4())
        profile = ProfileRecord(
            profile_id=profile_id,
            timestamp=datetime.now(),
            profile_type=ProfileType.BLOCK,
            metric_type=ProfileMetric.EXECUTION_TIME,
            operation=operation,
            value=execution_time,
            unit='milliseconds'
        )
        
        profiler.profile_queue.put(profile)

@contextmanager
def profile_memory(profiler: PerformanceProfiler, operation: str = None):
    """Context manager for profiling memory usage"""
    initial_snapshot = profiler.memory_profiler.take_snapshot()
    operation = operation or 'unknown_memory_operation'
    
    try:
        yield
    finally:
        final_snapshot = profiler.memory_profiler.take_snapshot()
        memory_diff = profiler.memory_profiler.compare_snapshots(initial_snapshot, final_snapshot)
        
        profile_id = str(uuid.uuid4())
        profile = ProfileRecord(
            profile_id=profile_id,
            timestamp=datetime.now(),
            profile_type=ProfileType.MEMORY,
            metric_type=ProfileMetric.MEMORY_USAGE,
            operation=operation,
            value=memory_diff.get('process_memory_diff', {}).get('rss_diff', 0),
            unit='bytes'
        )
        
        profiler.profile_queue.put(profile)

# Decorators for automatic profiling
def profile_function_calls(profiler: PerformanceProfiler):
    """Decorator to automatically profile function calls"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return profiler.profile_function(func, func.__name__, *args, **kwargs)
        return wrapper
    return decorator

def profile_memory_usage(profiler: PerformanceProfiler):
    """Decorator to automatically profile memory usage"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return profiler.profile_memory_operation(func.__name__, func, *args, **kwargs)
        return wrapper
    return decorator

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("PerformanceProfiler module loaded successfully")
    print("Use with DataStorage instance for full functionality")
