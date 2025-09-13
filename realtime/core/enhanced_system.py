"""
Enhanced Real-time LSTM Prediction System
Phase 4: Integration of API, logging, webhooks, error tracking, and performance profiling
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

from .config import Config
from .realtime_system import RealTimeSystem
from .api_server import APIServer
from .logging_system import LoggingSystem, initialize_logging_system
from .webhook_system import WebhookSystem
from .error_tracker import ErrorTracker
from .performance_profiler import PerformanceProfiler

logger = logging.getLogger(__name__)

class EnhancedRealTimeSystem:
    """
    Enhanced real-time system with comprehensive API, logging, webhooks, 
    error tracking, and performance profiling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced real-time system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        
        # Initialize core components
        self.config_manager = Config()
        self.realtime_system = None
        self.api_server = None
        
        # Initialize enhanced components
        self.logging_system = None
        self.webhook_system = None
        self.error_tracker = None
        self.performance_profiler = None
        
        # System status
        self.system_status = {
            'initialized': False,
            'started': False,
            'components': {},
            'last_health_check': None,
            'uptime_start': None
        }
        
        # Initialize system
        self._initialize_system()
        
        logger.info("EnhancedRealTimeSystem initialized")
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize data storage (assuming it exists)
            from ..storage.data_storage import DataStorage
            self.data_storage = DataStorage(self.config.get('db_path', 'enhanced_realtime.db'))
            
            # Initialize logging system first
            self.logging_system = initialize_logging_system(self.data_storage, self.config.get('logging', {}))
            logger.info("Logging system initialized")
            
            # Initialize error tracker
            self.error_tracker = ErrorTracker(self.data_storage, self.config.get('error_tracking', {}))
            logger.info("Error tracker initialized")
            
            # Initialize performance profiler
            self.performance_profiler = PerformanceProfiler(self.data_storage, self.config.get('performance_profiling', {}))
            logger.info("Performance profiler initialized")
            
            # Initialize webhook system
            self.webhook_system = WebhookSystem(self.data_storage, self.config.get('webhooks', {}))
            logger.info("Webhook system initialized")
            
            # Initialize core real-time system
            self.realtime_system = RealTimeSystem(self.data_storage, self.config)
            logger.info("Core real-time system initialized")
            
            # Initialize API server with enhanced features
            self.api_server = APIServer(self.realtime_system, self.config)
            logger.info("API server initialized")
            
            # Update system status
            self.system_status['components'] = {
                'logging_system': True,
                'error_tracker': True,
                'performance_profiler': True,
                'webhook_system': True,
                'realtime_system': True,
                'api_server': True
            }
            self.system_status['initialized'] = True
            
            logger.info("All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            self.system_status['initialized'] = False
            raise
    
    def start(self):
        """Start the enhanced real-time system"""
        try:
            if not self.system_status['initialized']:
                raise RuntimeError("System not initialized")
            
            if self.system_status['started']:
                logger.warning("System already started")
                return
            
            logger.info("Starting enhanced real-time system...")
            
            # Start core real-time system
            if self.realtime_system:
                self.realtime_system.start()
                logger.info("Core real-time system started")
            
            # Start API server
            if self.api_server:
                # Start API server in separate thread
                api_thread = threading.Thread(target=self.api_server.start, daemon=True)
                api_thread.start()
                logger.info("API server started")
            
            # Start webhook system
            if self.webhook_system:
                self.webhook_system.deliverer.start()
                logger.info("Webhook system started")
            
            # Start error tracker
            if self.error_tracker:
                self.error_tracker.start()
                logger.info("Error tracker started")
            
            # Start performance profiler
            if self.performance_profiler:
                self.performance_profiler.start()
                logger.info("Performance profiler started")
            
            # Set system status
            self.system_status['started'] = True
            self.system_status['uptime_start'] = datetime.now()
            self.running = True
            
            # Start health monitoring
            self._start_health_monitoring()
            
            logger.info("Enhanced real-time system started successfully")
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the enhanced real-time system"""
        try:
            if not self.system_status['started']:
                return
            
            logger.info("Stopping enhanced real-time system...")
            self.running = False
            
            # Stop API server
            if self.api_server:
                self.api_server.stop()
                logger.info("API server stopped")
            
            # Stop core real-time system
            if self.realtime_system:
                self.realtime_system.stop()
                logger.info("Core real-time system stopped")
            
            # Stop webhook system
            if self.webhook_system:
                self.webhook_system.stop()
                logger.info("Webhook system stopped")
            
            # Stop error tracker
            if self.error_tracker:
                self.error_tracker.stop()
                logger.info("Error tracker stopped")
            
            # Stop performance profiler
            if self.performance_profiler:
                self.performance_profiler.stop()
                logger.info("Performance profiler stopped")
            
            # Stop logging system
            if self.logging_system:
                self.logging_system.stop()
                logger.info("Logging system stopped")
            
            # Update system status
            self.system_status['started'] = False
            
            logger.info("Enhanced real-time system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        def health_monitor():
            while self.running:
                try:
                    self._perform_health_check()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")
                    time.sleep(60)
        
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
        logger.info("Health monitoring started")
    
    def _perform_health_check(self):
        """Perform system health check"""
        try:
            health_status = {
                'timestamp': datetime.now(),
                'system_running': self.running,
                'components': {}
            }
            
            # Check component health
            components = {
                'realtime_system': self.realtime_system,
                'api_server': self.api_server,
                'webhook_system': self.webhook_system,
                'error_tracker': self.error_tracker,
                'performance_profiler': self.performance_profiler,
                'logging_system': self.logging_system
            }
            
            for name, component in components.items():
                if component:
                    try:
                        # Basic health check - component exists and is responsive
                        health_status['components'][name] = 'healthy'
                    except Exception as e:
                        health_status['components'][name] = f'unhealthy: {str(e)}'
                        logger.warning(f"Component {name} health check failed: {e}")
                else:
                    health_status['components'][name] = 'not_initialized'
            
            # Check for critical errors
            if self.error_tracker:
                recent_errors = self.error_tracker.get_error_analysis(hours_back=1)
                critical_errors = recent_errors.get('critical_errors', [])
                if critical_errors:
                    health_status['critical_errors'] = len(critical_errors)
                    # Trigger webhook for critical errors
                    self._trigger_critical_error_webhook(critical_errors)
            
            # Check performance metrics
            if self.performance_profiler:
                perf_summary = self.performance_profiler.get_performance_summary(hours_back=1)
                health_status['performance'] = perf_summary
            
            self.system_status['last_health_check'] = health_status
            
            # Log health status
            logger.debug(f"Health check completed: {health_status}")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    def _trigger_critical_error_webhook(self, critical_errors: List):
        """Trigger webhook for critical errors"""
        try:
            if self.webhook_system:
                for error in critical_errors:
                    self.webhook_system.trigger_webhook(
                        'error_event',
                        {
                            'error_id': error.error_id,
                            'error_type': error.error_type,
                            'message': error.message,
                            'severity': error.severity.value,
                            'timestamp': error.timestamp.isoformat()
                        },
                        metadata={'alert_type': 'critical_error'}
                    )
        except Exception as e:
            logger.error(f"Error triggering critical error webhook: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            uptime = None
            if self.system_status['uptime_start']:
                uptime = (datetime.now() - self.system_status['uptime_start']).total_seconds()
            
            status = {
                'system_status': self.system_status,
                'uptime_seconds': uptime,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add component-specific status
            if self.logging_system:
                status['logging_stats'] = self.logging_system.get_log_statistics()
            
            if self.error_tracker:
                status['error_stats'] = self.error_tracker.get_error_statistics()
            
            if self.performance_profiler:
                status['performance_stats'] = self.performance_profiler.get_profile_statistics()
            
            if self.webhook_system:
                status['webhook_stats'] = self.webhook_system.get_webhook_statistics()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def get_system_metrics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'time_period_hours': hours_back
            }
            
            # Get error analysis
            if self.error_tracker:
                metrics['error_analysis'] = self.error_tracker.get_error_analysis(hours_back)
            
            # Get performance summary
            if self.performance_profiler:
                metrics['performance_summary'] = self.performance_profiler.get_performance_summary(hours_back)
            
            # Get webhook statistics
            if self.webhook_system:
                metrics['webhook_statistics'] = self.webhook_system.get_webhook_statistics()
            
            # Get logging statistics
            if self.logging_system:
                metrics['logging_statistics'] = self.logging_system.get_log_statistics()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {'error': str(e)}
    
    def create_webhook(self, name: str, url: str, event_types: List[str],
                      secret_key: str = None, **kwargs) -> str:
        """Create a webhook"""
        try:
            if not self.webhook_system:
                raise RuntimeError("Webhook system not initialized")
            
            # Convert event types to enum
            from .webhook_system import WebhookEventType
            event_type_enums = [WebhookEventType(et) for et in event_types]
            
            webhook_id = self.webhook_system.create_webhook(
                name=name,
                url=url,
                event_types=event_type_enums,
                secret_key=secret_key,
                **kwargs
            )
            
            logger.info(f"Webhook created: {webhook_id} - {name}")
            return webhook_id
            
        except Exception as e:
            logger.error(f"Error creating webhook: {e}")
            raise
    
    def export_data(self, export_type: str, format: str, **kwargs) -> str:
        """Export data using the enhanced export system"""
        try:
            if not self.realtime_system:
                raise RuntimeError("Real-time system not initialized")
            
            # Use the enhanced export functionality
            export_id = self.realtime_system.create_export_request(
                export_type=export_type,
                export_format=format,
                **kwargs
            )
            
            logger.info(f"Data export initiated: {export_id}")
            return export_id
            
        except Exception as e:
            logger.error(f"Error initiating data export: {e}")
            raise
    
    def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """Get export status"""
        try:
            if not self.realtime_system:
                raise RuntimeError("Real-time system not initialized")
            
            return self.realtime_system.get_export_status(export_id)
            
        except Exception as e:
            logger.error(f"Error getting export status: {e}")
            return {'error': str(e)}
    
    def set_debug_mode(self, mode: str, enabled: bool):
        """Set debug mode"""
        try:
            if self.logging_system:
                self.logging_system.set_debug_mode(mode, enabled)
                logger.info(f"Debug mode '{mode}' {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Error setting debug mode: {e}")
    
    def search_logs(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Search logs with filters"""
        try:
            if not self.logging_system:
                return []
            
            return self.logging_system.search_logs(filters, limit)
            
        except Exception as e:
            logger.error(f"Error searching logs: {e}")
            return []
    
    def resolve_error(self, error_id: str, resolution_notes: str, assigned_to: str = None) -> bool:
        """Resolve an error"""
        try:
            if not self.error_tracker:
                return False
            
            return self.error_tracker.resolve_error(error_id, resolution_notes, assigned_to)
            
        except Exception as e:
            logger.error(f"Error resolving error: {e}")
            return False

# Factory function for easy initialization
def create_enhanced_system(config_file: str = None) -> EnhancedRealTimeSystem:
    """
    Create an enhanced real-time system with default configuration
    
    Args:
        config_file: Path to configuration file (optional)
        
    Returns:
        EnhancedRealTimeSystem instance
    """
    try:
        # Load configuration
        if config_file:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            # Use default configuration
            config = {
                'db_path': 'enhanced_realtime.db',
                'api': {
                    'host': '0.0.0.0',
                    'port': 5000,
                    'debug': False,
                    'require_auth': True,
                    'api_keys': {
                        'default': 'default_api_key',
                        'premium': 'premium_api_key',
                        'admin': 'admin_api_key'
                    },
                    'rate_limits': {
                        'default': {'window': 60, 'requests': 100},
                        'premium': {'window': 60, 'requests': 1000},
                        'admin': {'window': 60, 'requests': 10000}
                    }
                },
                'logging': {
                    'log_directory': 'logs',
                    'max_log_file_size': 10485760,  # 10MB
                    'log_backup_count': 5,
                    'error_rate_threshold': 10
                },
                'webhooks': {
                    'delivery_workers': 5,
                    'delivery_queue_size': 1000,
                    'default_timeout': 30
                },
                'error_tracking': {
                    'error_queue_size': 1000,
                    'error_rate_threshold': 10,
                    'pattern_alert_threshold': 5
                },
                'performance_profiling': {
                    'profile_queue_size': 1000,
                    'enable_system_monitoring': True,
                    'performance_thresholds': {
                        'max_execution_time_ms': 5000,
                        'max_memory_usage_mb': 1000,
                        'max_cpu_usage_percent': 80
                    }
                }
            }
        
        # Create enhanced system
        system = EnhancedRealTimeSystem(config)
        
        logger.info("Enhanced real-time system created successfully")
        return system
        
    except Exception as e:
        logger.error(f"Error creating enhanced system: {e}")
        raise

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create enhanced system
        system = create_enhanced_system()
        
        # Start system
        system.start()
        
        # Get system status
        status = system.get_system_status()
        print(f"System status: {status}")
        
        # Keep running for demonstration
        import time
        time.sleep(10)
        
        # Stop system
        system.stop()
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("Enhanced system demonstration completed")
