"""
Error Tracking and Reporting System for Real-time LSTM Prediction System
Phase 4.3: Comprehensive error tracking, reporting, and alerting
"""

import logging
import traceback
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque
import queue
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories"""
    SYSTEM = "system"
    PREDICTION = "prediction"
    VALIDATION = "validation"
    API = "api"
    DATABASE = "database"
    MODEL = "model"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    EXTERNAL = "external"

class ErrorStatus(Enum):
    """Error status"""
    NEW = "new"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    IGNORED = "ignored"
    ESCALATED = "escalated"

@dataclass
class ErrorContext:
    """Error context information"""
    symbol: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ErrorRecord:
    """Error record"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: str
    message: str
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[ErrorContext] = None
    status: ErrorStatus = ErrorStatus.NEW
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    tags: List[str] = None
    related_errors: List[str] = None
    occurrence_count: int = 1
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.related_errors is None:
            self.related_errors = []
        if self.first_occurrence is None:
            self.first_occurrence = self.timestamp
        if self.last_occurrence is None:
            self.last_occurrence = self.timestamp

@dataclass
class ErrorPattern:
    """Error pattern for grouping similar errors"""
    pattern_id: str
    pattern_name: str
    error_type: str
    message_pattern: str
    stack_trace_pattern: str
    category: ErrorCategory
    severity: ErrorSeverity
    created_at: datetime
    error_count: int = 0
    last_seen: Optional[datetime] = None

class ErrorAggregator:
    """Aggregate and group similar errors"""
    
    def __init__(self):
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_groups: Dict[str, List[str]] = defaultdict(list)
        
    def add_error(self, error: ErrorRecord) -> str:
        """Add error and return pattern ID if matched"""
        pattern_id = self._find_matching_pattern(error)
        
        if pattern_id:
            # Update existing pattern
            pattern = self.error_patterns[pattern_id]
            pattern.error_count += 1
            pattern.last_seen = error.timestamp
            
            # Add to group
            self.error_groups[pattern_id].append(error.error_id)
            
            return pattern_id
        else:
            # Create new pattern
            pattern_id = self._create_pattern(error)
            self.error_groups[pattern_id] = [error.error_id]
            
            return pattern_id
    
    def _find_matching_pattern(self, error: ErrorRecord) -> Optional[str]:
        """Find matching error pattern"""
        for pattern_id, pattern in self.error_patterns.items():
            if self._matches_pattern(error, pattern):
                return pattern_id
        
        return None
    
    def _matches_pattern(self, error: ErrorRecord, pattern: ErrorPattern) -> bool:
        """Check if error matches pattern"""
        # Check error type
        if pattern.error_type != error.error_type:
            return False
        
        # Check message pattern (simplified matching)
        if pattern.message_pattern and pattern.message_pattern not in error.message:
            return False
        
        # Check stack trace pattern (simplified matching)
        if pattern.stack_trace_pattern and error.stack_trace:
            if pattern.stack_trace_pattern not in error.stack_trace:
                return False
        
        # Check category and severity
        if pattern.category != error.category or pattern.severity != error.severity:
            return False
        
        return True
    
    def _create_pattern(self, error: ErrorRecord) -> str:
        """Create new error pattern"""
        pattern_id = str(uuid.uuid4())
        
        pattern = ErrorPattern(
            pattern_id=pattern_id,
            pattern_name=f"Pattern_{error.error_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            error_type=error.error_type,
            message_pattern=error.message[:100] if error.message else "",
            stack_trace_pattern=error.stack_trace[:200] if error.stack_trace else "",
            category=error.category,
            severity=error.severity,
            created_at=datetime.now(),
            error_count=1,
            last_seen=error.timestamp
        )
        
        self.error_patterns[pattern_id] = pattern
        return pattern_id
    
    def get_pattern(self, pattern_id: str) -> Optional[ErrorPattern]:
        """Get error pattern by ID"""
        return self.error_patterns.get(pattern_id)
    
    def get_patterns(self) -> List[ErrorPattern]:
        """Get all error patterns"""
        return list(self.error_patterns.values())
    
    def get_errors_for_pattern(self, pattern_id: str) -> List[str]:
        """Get error IDs for a pattern"""
        return self.error_groups.get(pattern_id, [])

class ErrorAnalyzer:
    """Analyze error patterns and trends"""
    
    def __init__(self):
        self.error_trends = defaultdict(list)
        self.error_rates = defaultdict(float)
        
    def analyze_errors(self, errors: List[ErrorRecord], hours_back: int = 24) -> Dict[str, Any]:
        """Analyze errors for trends and patterns"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_errors = [e for e in errors if e.timestamp >= cutoff_time]
        
        analysis = {
            'total_errors': len(recent_errors),
            'errors_by_severity': self._count_by_severity(recent_errors),
            'errors_by_category': self._count_by_category(recent_errors),
            'top_error_types': self._get_top_error_types(recent_errors),
            'error_trends': self._analyze_trends(recent_errors),
            'error_rate': self._calculate_error_rate(recent_errors, hours_back),
            'critical_errors': [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL],
            'unresolved_errors': [e for e in recent_errors if e.status != ErrorStatus.RESOLVED]
        }
        
        return analysis
    
    def _count_by_severity(self, errors: List[ErrorRecord]) -> Dict[str, int]:
        """Count errors by severity"""
        counts = defaultdict(int)
        for error in errors:
            counts[error.severity.value] += 1
        return dict(counts)
    
    def _count_by_category(self, errors: List[ErrorRecord]) -> Dict[str, int]:
        """Count errors by category"""
        counts = defaultdict(int)
        for error in errors:
            counts[error.category.value] += 1
        return dict(counts)
    
    def _get_top_error_types(self, errors: List[ErrorRecord], top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top error types"""
        type_counts = defaultdict(int)
        for error in errors:
            type_counts[error.error_type] += 1
        
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'error_type': error_type, 'count': count} for error_type, count in sorted_types[:top_n]]
    
    def _analyze_trends(self, errors: List[ErrorRecord]) -> Dict[str, Any]:
        """Analyze error trends over time"""
        if not errors:
            return {}
        
        # Group by hour
        hourly_counts = defaultdict(int)
        for error in errors:
            hour_key = error.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
        
        # Calculate trend
        hours = sorted(hourly_counts.keys())
        if len(hours) < 2:
            return {'trend': 'insufficient_data'}
        
        counts = [hourly_counts[h] for h in hours]
        
        # Simple trend calculation
        if counts[-1] > counts[0]:
            trend = 'increasing'
        elif counts[-1] < counts[0]:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'hourly_counts': dict(hourly_counts),
            'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None
        }
    
    def _calculate_error_rate(self, errors: List[ErrorRecord], hours_back: int) -> float:
        """Calculate error rate per hour"""
        if hours_back == 0:
            return 0.0
        
        return len(errors) / hours_back

class ErrorReporter:
    """Generate error reports and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.report_templates = self._load_report_templates()
        
    def _load_report_templates(self) -> Dict[str, str]:
        """Load error report templates"""
        return {
            'daily_summary': """
Daily Error Summary - {date}
==============================

Total Errors: {total_errors}
Critical Errors: {critical_errors}
Unresolved Errors: {unresolved_errors}

Top Error Types:
{top_error_types}

Error Trends: {trend}

Critical Issues Requiring Attention:
{critical_issues}
""",
            'critical_alert': """
CRITICAL ERROR ALERT
===================

Error Type: {error_type}
Severity: {severity}
Category: {category}
Message: {message}
Timestamp: {timestamp}
Symbol: {symbol}

Stack Trace:
{stack_trace}

This error requires immediate attention.
""",
            'error_pattern_report': """
Error Pattern Analysis
=====================

Pattern: {pattern_name}
Error Type: {error_type}
Occurrences: {error_count}
First Seen: {first_seen}
Last Seen: {last_seen}

This pattern has occurred {error_count} times and may indicate a systemic issue.
"""
        }
    
    def generate_daily_report(self, analysis: Dict[str, Any]) -> str:
        """Generate daily error summary report"""
        template = self.report_templates['daily_summary']
        
        # Format top error types
        top_types = "\n".join([
            f"- {item['error_type']}: {item['count']} occurrences"
            for item in analysis.get('top_error_types', [])
        ])
        
        # Format critical issues
        critical_issues = "\n".join([
            f"- {error.error_type}: {error.message}"
            for error in analysis.get('critical_errors', [])
        ])
        
        return template.format(
            date=datetime.now().strftime('%Y-%m-%d'),
            total_errors=analysis.get('total_errors', 0),
            critical_errors=analysis.get('errors_by_severity', {}).get('critical', 0),
            unresolved_errors=len(analysis.get('unresolved_errors', [])),
            top_error_types=top_types,
            trend=analysis.get('error_trends', {}).get('trend', 'unknown'),
            critical_issues=critical_issues
        )
    
    def generate_critical_alert(self, error: ErrorRecord) -> str:
        """Generate critical error alert"""
        template = self.report_templates['critical_alert']
        
        return template.format(
            error_type=error.error_type,
            severity=error.severity.value,
            category=error.category.value,
            message=error.message,
            timestamp=error.timestamp.isoformat(),
            symbol=error.context.symbol if error.context else 'N/A',
            stack_trace=error.stack_trace or 'No stack trace available'
        )
    
    def generate_pattern_report(self, pattern: ErrorPattern) -> str:
        """Generate error pattern report"""
        template = self.report_templates['error_pattern_report']
        
        return template.format(
            pattern_name=pattern.pattern_name,
            error_type=pattern.error_type,
            error_count=pattern.error_count,
            first_seen=pattern.created_at.isoformat(),
            last_seen=pattern.last_seen.isoformat() if pattern.last_seen else 'N/A'
        )

class ErrorTracker:
    """Main error tracking system"""
    
    def __init__(self, data_storage, config: Dict[str, Any]):
        self.data_storage = data_storage
        self.config = config
        
        # Components
        self.aggregator = ErrorAggregator()
        self.analyzer = ErrorAnalyzer()
        self.reporter = ErrorReporter(config)
        
        # Error storage
        self.errors: Dict[str, ErrorRecord] = {}
        self.error_queue = queue.Queue(maxsize=config.get('error_queue_size', 1000))
        
        # Processing
        self.running = False
        self.processor_thread = None
        
        # Alerts and notifications
        self.alert_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'total_errors': 0,
            'errors_by_severity': defaultdict(int),
            'errors_by_category': defaultdict(int),
            'last_error': None
        }
        
        # Start processing
        self.start()
        
        logger.info("ErrorTracker initialized")
    
    def start(self):
        """Start error tracking"""
        if self.running:
            return
        
        self.running = True
        self.processor_thread = threading.Thread(target=self._process_errors, daemon=True)
        self.processor_thread.start()
        
        logger.info("ErrorTracker started")
    
    def stop(self):
        """Stop error tracking"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        
        logger.info("ErrorTracker stopped")
    
    def track_error(self, exception: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                   category: ErrorCategory = ErrorCategory.SYSTEM, context: ErrorContext = None,
                   tags: List[str] = None) -> str:
        """Track an error"""
        try:
            error_id = str(uuid.uuid4())
            
            # Extract error information
            error_type = type(exception).__name__
            message = str(exception)
            stack_trace = traceback.format_exc()
            
            # Create error record
            error = ErrorRecord(
                error_id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                error_type=error_type,
                message=message,
                exception=message,
                stack_trace=stack_trace,
                context=context,
                tags=tags or []
            )
            
            # Queue for processing
            self.error_queue.put(error)
            
            logger.error(f"Error tracked: {error_id} - {error_type}: {message}")
            return error_id
            
        except Exception as e:
            logger.error(f"Error tracking error: {e}")
            return ""
    
    def _process_errors(self):
        """Process errors from queue"""
        while self.running:
            try:
                # Get error from queue
                error = self.error_queue.get(timeout=1)
                
                # Store error
                self.errors[error.error_id] = error
                self._store_error(error)
                
                # Aggregate error
                pattern_id = self.aggregator.add_error(error)
                
                # Update statistics
                self._update_stats(error)
                
                # Check for alerts
                self._check_alerts(error)
                
                self.error_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing error: {e}")
    
    def _store_error(self, error: ErrorRecord):
        """Store error in database"""
        try:
            self.data_storage.store_error_record(
                error_id=error.error_id,
                timestamp=error.timestamp,
                severity=error.severity.value,
                category=error.category.value,
                error_type=error.error_type,
                message=error.message,
                exception=error.exception,
                stack_trace=error.stack_trace,
                context=json.dumps(asdict(error.context)) if error.context else None,
                status=error.status.value,
                assigned_to=error.assigned_to,
                resolution_notes=error.resolution_notes,
                resolved_at=error.resolved_at,
                tags=json.dumps(error.tags),
                related_errors=json.dumps(error.related_errors),
                occurrence_count=error.occurrence_count,
                first_occurrence=error.first_occurrence,
                last_occurrence=error.last_occurrence
            )
        except Exception as e:
            logger.error(f"Error storing error record: {e}")
    
    def _update_stats(self, error: ErrorRecord):
        """Update error statistics"""
        self.stats['total_errors'] += 1
        self.stats['errors_by_severity'][error.severity.value] += 1
        self.stats['errors_by_category'][error.category.value] += 1
        self.stats['last_error'] = error.timestamp
    
    def _check_alerts(self, error: ErrorRecord):
        """Check for error-based alerts"""
        try:
            # Critical error alert
            if error.severity == ErrorSeverity.CRITICAL:
                self._send_critical_alert(error)
            
            # High error rate alert
            self._check_error_rate_alert()
            
            # Pattern-based alerts
            self._check_pattern_alerts(error)
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _send_critical_alert(self, error: ErrorRecord):
        """Send critical error alert"""
        try:
            alert_message = self.reporter.generate_critical_alert(error)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(error, alert_message)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            # Store alert
            self.data_storage.store_system_alert(
                alert_type='CRITICAL_ERROR',
                severity='CRITICAL',
                message=alert_message,
                details={
                    'error_id': error.error_id,
                    'error_type': error.error_type,
                    'category': error.category.value
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending critical alert: {e}")
    
    def _check_error_rate_alert(self):
        """Check for high error rate"""
        try:
            # Count errors in last 5 minutes
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            recent_errors = [
                e for e in self.errors.values()
                if e.timestamp >= five_minutes_ago
            ]
            
            threshold = self.config.get('error_rate_threshold', 10)
            if len(recent_errors) > threshold:
                self._send_error_rate_alert(len(recent_errors), threshold)
                
        except Exception as e:
            logger.error(f"Error checking error rate: {e}")
    
    def _send_error_rate_alert(self, error_count: int, threshold: int):
        """Send error rate alert"""
        try:
            message = f"High error rate detected: {error_count} errors in 5 minutes (threshold: {threshold})"
            
            self.data_storage.store_system_alert(
                alert_type='HIGH_ERROR_RATE',
                severity='HIGH',
                message=message,
                details={
                    'error_count': error_count,
                    'threshold': threshold,
                    'time_window': '5 minutes'
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending error rate alert: {e}")
    
    def _check_pattern_alerts(self, error: ErrorRecord):
        """Check for pattern-based alerts"""
        try:
            # Get pattern for this error
            pattern_id = None
            for pid, pattern in self.aggregator.error_patterns.items():
                if self.aggregator._matches_pattern(error, pattern):
                    pattern_id = pid
                    break
            
            if pattern_id:
                pattern = self.aggregator.error_patterns[pattern_id]
                
                # Alert if pattern has high occurrence count
                threshold = self.config.get('pattern_alert_threshold', 5)
                if pattern.error_count >= threshold:
                    self._send_pattern_alert(pattern)
                    
        except Exception as e:
            logger.error(f"Error checking pattern alerts: {e}")
    
    def _send_pattern_alert(self, pattern: ErrorPattern):
        """Send pattern-based alert"""
        try:
            alert_message = self.reporter.generate_pattern_report(pattern)
            
            self.data_storage.store_system_alert(
                alert_type='ERROR_PATTERN',
                severity='MEDIUM',
                message=alert_message,
                details={
                    'pattern_id': pattern.pattern_id,
                    'error_type': pattern.error_type,
                    'occurrence_count': pattern.error_count
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending pattern alert: {e}")
    
    def get_error_analysis(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get error analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_errors = [e for e in self.errors.values() if e.timestamp >= cutoff_time]
        
        return self.analyzer.analyze_errors(recent_errors, hours_back)
    
    def get_error_patterns(self) -> List[ErrorPattern]:
        """Get error patterns"""
        return self.aggregator.get_patterns()
    
    def resolve_error(self, error_id: str, resolution_notes: str, assigned_to: str = None) -> bool:
        """Resolve an error"""
        try:
            if error_id not in self.errors:
                return False
            
            error = self.errors[error_id]
            error.status = ErrorStatus.RESOLVED
            error.resolution_notes = resolution_notes
            error.resolved_at = datetime.now()
            if assigned_to:
                error.assigned_to = assigned_to
            
            # Update in database
            self._store_error(error)
            
            logger.info(f"Error resolved: {error_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving error: {e}")
            return False
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            'stats': self.stats,
            'total_patterns': len(self.aggregator.error_patterns),
            'queue_size': self.error_queue.qsize(),
            'running': self.running
        }

# Context manager for error tracking
@contextmanager
def track_errors(error_tracker: ErrorTracker, category: ErrorCategory = ErrorCategory.SYSTEM,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM, **context_kwargs):
    """Context manager for tracking errors in a block of code"""
    try:
        yield
    except Exception as e:
        context = ErrorContext(**context_kwargs)
        error_tracker.track_error(e, severity=severity, category=category, context=context)
        raise

# Decorator for automatic error tracking
def track_function_errors(error_tracker: ErrorTracker, 
                         category: ErrorCategory = ErrorCategory.SYSTEM,
                         severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator to automatically track errors in functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create context from function info
                context = ErrorContext(
                    module=func.__module__,
                    function=func.__name__,
                    operation=f"function_call_{func.__name__}"
                )
                
                error_tracker.track_error(e, severity=severity, category=category, context=context)
                raise
        
        return wrapper
    return decorator

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("ErrorTracker module loaded successfully")
    print("Use with DataStorage instance for full functionality")
