"""
Comprehensive Logging System for Real-time LSTM Prediction System
Phase 4.3: Advanced logging, debugging, and error tracking
"""

import logging
import logging.handlers

logger = logging.getLogger(__name__)
import json
import traceback
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import queue
import uuid
from contextlib import contextmanager
import functools
import inspect

class LogLevel(Enum):
    """Log levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LogCategory(Enum):
    """Log categories"""
    SYSTEM = "system"
    PREDICTION = "prediction"
    VALIDATION = "validation"
    API = "api"
    DATABASE = "database"
    MODEL = "model"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"

@dataclass
class LogEntry:
    """Structured log entry"""
    log_id: str
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    symbol: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = None
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class LogHandler:
    """Custom log handler with structured logging"""
    
    def __init__(self, log_queue: queue.Queue):
        self.log_queue = log_queue
        self.level = logging.NOTSET
    
    def setLevel(self, level):
        """Set the logging level for this handler"""
        self.level = level
    
    def getLevel(self):
        """Get the logging level for this handler"""
        return self.level
    
    def emit(self, record):
        """Emit log record to queue"""
        try:
            # Extract structured information
            log_entry = LogEntry(
                log_id=str(uuid.uuid4()),
                timestamp=datetime.fromtimestamp(record.created),
                level=LogLevel(record.levelname),
                category=getattr(record, 'category', LogCategory.SYSTEM),
                message=record.getMessage(),
                module=record.module,
                function=record.funcName,
                line_number=record.lineno,
                thread_id=record.thread,
                process_id=record.process,
                symbol=getattr(record, 'symbol', None),
                user_id=getattr(record, 'user_id', None),
                session_id=getattr(record, 'session_id', None),
                request_id=getattr(record, 'request_id', None),
                duration_ms=getattr(record, 'duration_ms', None),
                metadata=getattr(record, 'metadata', {}),
                exception=record.exc_text,
                stack_trace=traceback.format_exc() if record.exc_info else None
            )
            
            self.log_queue.put(log_entry)
        except Exception as e:
            print(f"Error in log handler: {e}")

class LogProcessor:
    """Process and store log entries"""
    
    def __init__(self, data_storage, config: Dict[str, Any]):
        self.data_storage = data_storage
        self.config = config
        self.log_queue = queue.Queue(maxsize=config.get('log_queue_size', 10000))
        self.running = False
        self.processor_thread = None
        
        # Log statistics
        self.stats = {
            'total_logs': 0,
            'logs_by_level': {},
            'logs_by_category': {},
            'errors': 0,
            'last_processed': None
        }
        
        # Start processor
        self.start()
    
    def start(self):
        """Start log processor"""
        if self.running:
            return
        
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.processor_thread.start()
    
    def stop(self):
        """Stop log processor"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
    
    def _process_logs(self):
        """Process log entries from queue"""
        while self.running:
            try:
                # Get log entry with timeout
                log_entry = self.log_queue.get(timeout=1)
                
                # Store in database
                self._store_log_entry(log_entry)
                
                # Update statistics
                self._update_stats(log_entry)
                
                # Check for log-based alerts
                self._check_log_alerts(log_entry)
                
                self.log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing log entry: {e}")
    
    def _store_log_entry(self, log_entry: LogEntry):
        """Store log entry in database"""
        try:
            self.data_storage.store_log_entry(
                log_id=log_entry.log_id,
                timestamp=log_entry.timestamp,
                level=log_entry.level.value,
                category=log_entry.category.value,
                message=log_entry.message,
                module=log_entry.module,
                function=log_entry.function,
                line_number=log_entry.line_number,
                thread_id=log_entry.thread_id,
                process_id=log_entry.process_id,
                symbol=log_entry.symbol,
                user_id=log_entry.user_id,
                session_id=log_entry.session_id,
                request_id=log_entry.request_id,
                duration_ms=log_entry.duration_ms,
                metadata=json.dumps(log_entry.metadata),
                exception=log_entry.exception,
                stack_trace=log_entry.stack_trace
            )
        except Exception as e:
            print(f"Error storing log entry: {e}")
    
    def _update_stats(self, log_entry: LogEntry):
        """Update log statistics"""
        self.stats['total_logs'] += 1
        self.stats['last_processed'] = log_entry.timestamp
        
        # Update level stats
        level = log_entry.level.value
        self.stats['logs_by_level'][level] = self.stats['logs_by_level'].get(level, 0) + 1
        
        # Update category stats
        category = log_entry.category.value
        self.stats['logs_by_category'][category] = self.stats['logs_by_category'].get(category, 0) + 1
        
        # Update error count
        if log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.stats['errors'] += 1
    
    def _check_log_alerts(self, log_entry: LogEntry):
        """Check for log-based alerts"""
        try:
            # Check for error rate alerts
            if log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self._check_error_rate_alert()
            
            # Check for specific error patterns
            if 'database' in log_entry.message.lower() and log_entry.level == LogLevel.ERROR:
                self._create_database_error_alert(log_entry)
            
            # Check for security events
            if log_entry.category == LogCategory.SECURITY:
                self._create_security_alert(log_entry)
                
        except Exception as e:
            print(f"Error checking log alerts: {e}")
    
    def _check_error_rate_alert(self):
        """Check if error rate exceeds threshold"""
        try:
            # Count errors in last 5 minutes
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            error_count = self.data_storage.count_logs_by_level(
                level='ERROR',
                since=five_minutes_ago
            )
            
            threshold = self.config.get('error_rate_threshold', 10)
            if error_count > threshold:
                self._create_error_rate_alert(error_count, threshold)
                
        except Exception as e:
            print(f"Error checking error rate: {e}")
    
    def _create_database_error_alert(self, log_entry: LogEntry):
        """Create database error alert"""
        try:
            self.data_storage.store_system_alert(
                alert_type='DATABASE_ERROR',
                severity='HIGH',
                message=f'Database error detected: {log_entry.message}',
                details={
                    'log_id': log_entry.log_id,
                    'module': log_entry.module,
                    'function': log_entry.function,
                    'exception': log_entry.exception
                }
            )
        except Exception as e:
            print(f"Error creating database error alert: {e}")
    
    def _create_security_alert(self, log_entry: LogEntry):
        """Create security alert"""
        try:
            self.data_storage.store_system_alert(
                alert_type='SECURITY_EVENT',
                severity='CRITICAL',
                message=f'Security event detected: {log_entry.message}',
                details={
                    'log_id': log_entry.log_id,
                    'category': log_entry.category.value,
                    'user_id': log_entry.user_id,
                    'session_id': log_entry.session_id,
                    'metadata': log_entry.metadata
                }
            )
        except Exception as e:
            print(f"Error creating security alert: {e}")
    
    def _create_error_rate_alert(self, error_count: int, threshold: int):
        """Create error rate alert"""
        try:
            self.data_storage.store_system_alert(
                alert_type='HIGH_ERROR_RATE',
                severity='HIGH',
                message=f'High error rate detected: {error_count} errors in 5 minutes (threshold: {threshold})',
                details={
                    'error_count': error_count,
                    'threshold': threshold,
                    'time_window': '5 minutes'
                }
            )
        except Exception as e:
            print(f"Error creating error rate alert: {e}")

class LoggingSystem:
    """Comprehensive logging system"""
    
    def __init__(self, data_storage, config: Dict[str, Any]):
        self.data_storage = data_storage
        self.config = config
        
        # Setup log directories
        self.log_dir = Path(config.get('log_directory', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log processor
        self.log_processor = LogProcessor(data_storage, config)
        
        # Setup loggers
        self.loggers = {}
        self._setup_loggers()
        
        # Performance tracking
        self.performance_tracker = LogPerformanceTracker()
        
        # Debug modes
        self.debug_modes = {
            'detailed_timing': False,
            'sql_queries': False,
            'api_requests': False,
            'model_predictions': False,
            'data_flow': False
        }
        
        logger.info("LoggingSystem initialized")
    
    def _setup_loggers(self):
        """Setup loggers for different modules"""
        try:
            # Create log handler
            log_handler = LogHandler(self.log_processor.log_queue)
            log_handler.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            log_handler.setFormatter(formatter)
            
            # Setup root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            root_logger.addHandler(log_handler)
            
            # Setup file handlers for different categories
            self._setup_file_handlers()
            
            # Setup module-specific loggers
            modules = [
                'realtime.core',
                'realtime.models',
                'realtime.validation',
                'realtime.monitoring',
                'realtime.api',
                'realtime.database'
            ]
            
            for module in modules:
                logger = logging.getLogger(module)
                logger.setLevel(logging.DEBUG)
                logger.propagate = True
                self.loggers[module] = logger
            
        except Exception as e:
            print(f"Error setting up loggers: {e}")
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log categories"""
        try:
            # Rotating file handlers for different categories
            categories = [
                ('system', 'system.log'),
                ('prediction', 'prediction.log'),
                ('validation', 'validation.log'),
                ('api', 'api.log'),
                ('database', 'database.log'),
                ('performance', 'performance.log'),
                ('security', 'security.log')
            ]
            
            for category, filename in categories:
                file_path = self.log_dir / filename
                handler = logging.handlers.RotatingFileHandler(
                    file_path,
                    maxBytes=self.config.get('max_log_file_size', 10 * 1024 * 1024),  # 10MB
                    backupCount=self.config.get('log_backup_count', 5)
                )
                handler.setLevel(logging.DEBUG)
                
                # Custom formatter for structured logging
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
                )
                handler.setFormatter(formatter)
                
                # Add to root logger
                logging.getLogger().addHandler(handler)
                
        except Exception as e:
            print(f"Error setting up file handlers: {e}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger for specific module"""
        return logging.getLogger(name)
    
    def log_structured(self, level: LogLevel, category: LogCategory, message: str,
                      symbol: str = None, user_id: str = None, session_id: str = None,
                      request_id: str = None, duration_ms: float = None,
                      metadata: Dict[str, Any] = None, exception: Exception = None):
        """Log structured message"""
        try:
            # Get caller information
            frame = inspect.currentframe().f_back
            module = frame.f_globals.get('__name__', 'unknown')
            function = frame.f_code.co_name
            line_number = frame.f_lineno
            
            # Create log record
            record = logging.LogRecord(
                name=module,
                level=level.value,
                pathname=frame.f_code.co_filename,
                lineno=line_number,
                msg=message,
                args=(),
                exc_info=exception
            )
            
            # Add custom attributes
            record.category = category
            record.symbol = symbol
            record.user_id = user_id
            record.session_id = session_id
            record.request_id = request_id
            record.duration_ms = duration_ms
            record.metadata = metadata or {}
            
            # Log the record
            logger = logging.getLogger(module)
            logger.handle(record)
            
        except Exception as e:
            print(f"Error in structured logging: {e}")
    
    @contextmanager
    def log_performance(self, operation: str, symbol: str = None, metadata: Dict[str, Any] = None):
        """Context manager for performance logging"""
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        try:
            self.log_structured(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                f"Starting {operation}",
                symbol=symbol,
                metadata={
                    'operation': operation,
                    'operation_id': operation_id,
                    **(metadata or {})
                }
            )
            
            yield operation_id
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log_structured(
                LogLevel.ERROR,
                LogCategory.PERFORMANCE,
                f"Error in {operation}: {str(e)}",
                symbol=symbol,
                duration_ms=duration_ms,
                metadata={
                    'operation': operation,
                    'operation_id': operation_id,
                    'error': str(e),
                    **(metadata or {})
                },
                exception=e
            )
            raise
        else:
            duration_ms = (time.time() - start_time) * 1000
            self.log_structured(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                f"Completed {operation}",
                symbol=symbol,
                duration_ms=duration_ms,
                metadata={
                    'operation': operation,
                    'operation_id': operation_id,
                    **(metadata or {})
                }
            )
    
    def set_debug_mode(self, mode: str, enabled: bool):
        """Set debug mode"""
        if mode in self.debug_modes:
            self.debug_modes[mode] = enabled
            self.log_structured(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                f"Debug mode '{mode}' {'enabled' if enabled else 'disabled'}"
            )
    
    def is_debug_mode(self, mode: str) -> bool:
        """Check if debug mode is enabled"""
        return self.debug_modes.get(mode, False)
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get log statistics"""
        return {
            'processor_stats': self.log_processor.stats,
            'debug_modes': self.debug_modes,
            'log_directory': str(self.log_dir),
            'performance_stats': self.performance_tracker.get_stats()
        }
    
    def search_logs(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Search logs with filters"""
        try:
            return self.data_storage.search_logs(filters, limit)
        except Exception as e:
            self.log_structured(
                LogLevel.ERROR,
                LogCategory.DATABASE,
                f"Error searching logs: {str(e)}",
                exception=e
            )
            return []
    
    def cleanup_old_logs(self, days_back: int = 30) -> bool:
        """Clean up old logs"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            deleted_count = self.data_storage.cleanup_old_logs(cutoff_date)
            
            self.log_structured(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                f"Cleaned up {deleted_count} old log entries"
            )
            return True
            
        except Exception as e:
            self.log_structured(
                LogLevel.ERROR,
                LogCategory.SYSTEM,
                f"Error cleaning up old logs: {str(e)}",
                exception=e
            )
            return False
    
    def stop(self):
        """Stop logging system"""
        try:
            self.log_processor.stop()
            self.log_structured(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                "Logging system stopped"
            )
        except Exception as e:
            print(f"Error stopping logging system: {e}")

class LogPerformanceTracker:
    """Track performance metrics from logs"""
    
    def __init__(self):
        self.metrics = {
            'operation_times': {},
            'error_rates': {},
            'throughput': {},
            'resource_usage': {}
        }
    
    def track_operation(self, operation: str, duration_ms: float, success: bool = True):
        """Track operation performance"""
        if operation not in self.metrics['operation_times']:
            self.metrics['operation_times'][operation] = []
        
        self.metrics['operation_times'][operation].append({
            'duration_ms': duration_ms,
            'success': success,
            'timestamp': datetime.now()
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for operation, times in self.metrics['operation_times'].items():
            if times:
                durations = [t['duration_ms'] for t in times]
                successes = [t['success'] for t in times]
                
                stats[operation] = {
                    'avg_duration_ms': sum(durations) / len(durations),
                    'min_duration_ms': min(durations),
                    'max_duration_ms': max(durations),
                    'success_rate': sum(successes) / len(successes),
                    'total_operations': len(times)
                }
        
        return stats

# Decorators for automatic logging
def log_function_call(category: LogCategory = LogCategory.SYSTEM, log_args: bool = False):
    """Decorator to automatically log function calls"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logging system instance (assuming it's available globally)
            logging_system = getattr(logging_system, 'instance', None)
            if not logging_system:
                return func(*args, **kwargs)
            
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"
            
            # Log function entry
            if log_args:
                logging_system.log_structured(
                    LogLevel.DEBUG,
                    category,
                    f"Calling {function_name}",
                    metadata={'args': str(args), 'kwargs': str(kwargs)}
                )
            else:
                logging_system.log_structured(
                    LogLevel.DEBUG,
                    category,
                    f"Calling {function_name}"
                )
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log successful completion
                logging_system.log_structured(
                    LogLevel.DEBUG,
                    category,
                    f"Completed {function_name}",
                    duration_ms=duration_ms
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                # Log error
                logging_system.log_structured(
                    LogLevel.ERROR,
                    category,
                    f"Error in {function_name}: {str(e)}",
                    duration_ms=duration_ms,
                    exception=e
                )
                raise
        
        return wrapper
    return decorator

def log_api_request(category: LogCategory = LogCategory.API):
    """Decorator to log API requests"""
    return log_function_call(category, log_args=True)

def log_database_operation(category: LogCategory = LogCategory.DATABASE):
    """Decorator to log database operations"""
    return log_function_call(category)

def log_prediction_operation(category: LogCategory = LogCategory.PREDICTION):
    """Decorator to log prediction operations"""
    return log_function_call(category)

# Global logging system instance
logging_system = None

def initialize_logging_system(data_storage, config: Dict[str, Any]):
    """Initialize global logging system"""
    global logging_system
    logging_system = LoggingSystem(data_storage, config)
    logging_system.instance = logging_system
    return logging_system

def get_logging_system() -> LoggingSystem:
    """Get global logging system instance"""
    return logging_system

# Example usage and testing
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO)
    
    print("LoggingSystem module loaded successfully")
    print("Use with DataStorage instance for full functionality")
