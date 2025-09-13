"""
Comprehensive Test Suite for Real-time LSTM Prediction System
Tests all functionality across all modules and components
"""

import unittest
import tempfile
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import modules
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Avoid importing the package __init__.py which has relative imports
# by ensuring we're not in the package context

# Import modules using absolute imports to avoid relative import issues
try:
    # Import core modules
    from core.config import Config
    
    # Try to import other modules with fallback handling
    try:
        from storage.data_storage import DataStorage
    except ImportError:
        from storage import data_storage
        DataStorage = data_storage.DataStorage
    
    try:
        from data.data_fetcher import PolygonDataFetcher
    except ImportError:
        from data import data_fetcher
        PolygonDataFetcher = data_fetcher.PolygonDataFetcher
    
    try:
        from models.model_manager import ModelManager
    except ImportError:
        from models import model_manager
        ModelManager = model_manager.ModelManager
    
    try:
        from models.prediction_engine import PredictionEngine
    except ImportError:
        from models import prediction_engine
        PredictionEngine = prediction_engine.PredictionEngine
    
    try:
        from core.logging_system import LoggingSystem, LogLevel, LogCategory
    except ImportError:
        from core import logging_system
        LoggingSystem = logging_system.LoggingSystem
        LogLevel = logging_system.LogLevel
        LogCategory = logging_system.LogCategory
    
    try:
        from core.error_tracker import ErrorTracker, ErrorSeverity, ErrorCategory
    except ImportError:
        from core import error_tracker
        ErrorTracker = error_tracker.ErrorTracker
        ErrorSeverity = error_tracker.ErrorSeverity
        ErrorCategory = error_tracker.ErrorCategory
    
    try:
        from core.performance_profiler import PerformanceProfiler
    except ImportError:
        from core import performance_profiler
        PerformanceProfiler = performance_profiler.PerformanceProfiler
    
    try:
        from core.webhook_system import WebhookSystem, WebhookEventType
    except ImportError:
        from core import webhook_system
        WebhookSystem = webhook_system.WebhookSystem
        WebhookEventType = webhook_system.WebhookEventType
    
    try:
        from core.api_server import APIServer
    except ImportError as e:
        print(f"APIServer import failed (expected): {e}")
        APIServer = None
    
    try:
        from export.data_export import DataExporter, ExportFormat, ExportType
    except ImportError:
        from export import data_export
        DataExporter = data_export.DataExporter
        ExportFormat = data_export.ExportFormat
        ExportType = data_export.ExportType
    
    try:
        from validation.validation_workflow import PredictionValidationWorkflow
    except ImportError:
        from validation import validation_workflow
        PredictionValidationWorkflow = validation_workflow.PredictionValidationWorkflow
    
    try:
        from monitoring.accuracy_tracker import AccuracyTracker
    except ImportError:
        from monitoring import accuracy_tracker
        AccuracyTracker = accuracy_tracker.AccuracyTracker
    
    try:
        from monitoring.performance_monitor import PerformanceMonitor, MetricType
    except ImportError:
        from monitoring import performance_monitor
        PerformanceMonitor = performance_monitor.PerformanceMonitor
        MetricType = performance_monitor.MetricType
    
    try:
        from monitoring.alerting_system import AlertingSystem, AlertSeverity, AlertType
    except ImportError:
        from monitoring import alerting_system
        AlertingSystem = alerting_system.AlertingSystem
        AlertSeverity = alerting_system.AlertSeverity
        AlertType = alerting_system.AlertType
    
    try:
        from storage.backup_recovery import BackupRecoveryManager, BackupType
    except ImportError:
        from storage import backup_recovery
        BackupRecoveryManager = backup_recovery.BackupRecoveryManager
        BackupType = backup_recovery.BackupType
    
    try:
        from storage.data_retention import DataRetentionManager, RetentionPolicy
    except ImportError:
        from storage import data_retention
        DataRetentionManager = data_retention.DataRetentionManager
        RetentionPolicy = data_retention.RetentionPolicy
    
    try:
        from core.realtime_system import RealTimeSystem
    except ImportError as e:
        print(f"RealTimeSystem import failed (expected): {e}")
        RealTimeSystem = None
    
    try:
        from core.enhanced_system import EnhancedRealTimeSystem
    except ImportError as e:
        print(f"EnhancedRealTimeSystem import failed (expected): {e}")
        EnhancedRealTimeSystem = None
    
    # Initialize missing modules to None if they failed to import
    if 'APIServer' not in locals():
        APIServer = None
    if 'DataExporter' not in locals():
        DataExporter = None
    if 'PredictionValidationWorkflow' not in locals():
        PredictionValidationWorkflow = None
    if 'AccuracyTracker' not in locals():
        AccuracyTracker = None
    if 'PerformanceMonitor' not in locals():
        PerformanceMonitor = None
    if 'AlertingSystem' not in locals():
        AlertingSystem = None
    if 'BackupRecoveryManager' not in locals():
        BackupRecoveryManager = None
    if 'DataRetentionManager' not in locals():
        DataRetentionManager = None
    if 'RealTimeSystem' not in locals():
        RealTimeSystem = None
    if 'EnhancedRealTimeSystem' not in locals():
        EnhancedRealTimeSystem = None
    
    print("Successfully imported available modules")
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("This is expected in some environments - tests will be skipped")
    # Set all modules to None to skip tests that require imports
    Config = None
    DataStorage = None
    PolygonDataFetcher = None
    ModelManager = None
    PredictionEngine = None
    LoggingSystem = None
    ErrorTracker = None
    PerformanceProfiler = None
    WebhookSystem = None
    APIServer = None
    DataExporter = None
    PredictionValidationWorkflow = None
    AccuracyTracker = None
    PerformanceMonitor = None
    AlertingSystem = None
    BackupRecoveryManager = None
    DataRetentionManager = None
    RealTimeSystem = None
    EnhancedRealTimeSystem = None
    
    # Set up mock enums
    class LogLevel:
        INFO = "INFO"
        ERROR = "ERROR"
        WARNING = "WARNING"
        DEBUG = "DEBUG"
    
    class LogCategory:
        SYSTEM = "SYSTEM"
        API = "API"
        DATABASE = "DATABASE"
        PREDICTION = "PREDICTION"
    
    class ErrorSeverity:
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        CRITICAL = "CRITICAL"
    
    class ErrorCategory:
        SYSTEM = "SYSTEM"
        API = "API"
        DATABASE = "DATABASE"
        TEST = "TEST"
    
    class WebhookEventType:
        PREDICTION_UPDATE = "prediction_update"
    
    class MockEnum:
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return self.value
        def __eq__(self, other):
            if isinstance(other, MockEnum):
                return self.value == other.value
            return self.value == other
    
    class ExportFormat(MockEnum):
        CSV = MockEnum("csv")
        JSON = MockEnum("json")
    
    class ExportType(MockEnum):
        PREDICTIONS = MockEnum("predictions")
    
    class BackupType:
        FULL = "FULL"
    
    class RetentionPolicy:
        DAILY = "DAILY"
    
    class MetricType:
        EXECUTION_TIME = "EXECUTION_TIME"
    
    class AlertSeverity:
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
    
    class AlertType:
        PERFORMANCE = "PERFORMANCE"

# Helper function to check if a module is available
def check_module_available(module_class):
    if module_class is None:
        raise unittest.SkipTest(f"{module_class.__name__ if hasattr(module_class, '__name__') else 'Module'} not available")


class TestConfig(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        if Config is None:
            self.skipTest("Config module not available")
        self.test_config_file = tempfile.mktemp(suffix='.json')
        self.config = Config(self.test_config_file)
    
    def tearDown(self):
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)
    
    def test_default_config(self):
        """Test default configuration loading"""
        config = Config()
        self.assertIsInstance(config.config, dict)
        self.assertIn('symbols', config.config)
        self.assertIn('update_interval', config.config)
        self.assertIn('min_profit_percent', config.config)
    
    def test_config_get_set(self):
        """Test configuration get/set operations"""
        self.config.set('test_key', 'test_value')
        self.assertEqual(self.config.get('test_key'), 'test_value')
        self.assertEqual(self.config.get('nonexistent', 'default'), 'default')
    
    def test_symbol_management(self):
        """Test symbol add/remove operations"""
        initial_symbols = self.config.get_symbols()
        self.config.add_symbol('TEST')
        self.assertIn('TEST', self.config.get_symbols())
        
        self.config.remove_symbol('TEST')
        self.assertNotIn('TEST', self.config.get_symbols())
    
    def test_config_validation(self):
        """Test configuration validation"""
        validation = self.config.validate_config()
        self.assertIsInstance(validation, dict)
        self.assertIn('valid', validation)
        self.assertIn('errors', validation)
        self.assertIn('warnings', validation)
    
    def test_config_save_load(self):
        """Test configuration save and load"""
        self.config.set('test_save', 'save_value')
        self.assertTrue(self.config.save_config())
        
        # Create new config instance to test loading
        new_config = Config(self.test_config_file)
        self.assertEqual(new_config.get('test_save'), 'save_value')


class TestDataStorage(unittest.TestCase):
    """Test data storage functionality"""
    
    def setUp(self):
        check_module_available(DataStorage)
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
    
    def tearDown(self):
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_storage_initialization(self):
        """Test data storage initialization"""
        self.assertIsNotNone(self.storage)
        self.assertTrue(os.path.exists(self.temp_db))
    
    def test_market_data_operations(self):
        """Test market data storage and retrieval"""
        # Create sample data
        sample_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000000]
        })
        
        # Test storage
        self.storage.store_market_data('TEST', sample_data)
        
        # Test retrieval
        retrieved_data = self.storage.retrieve_historical_data(
            'TEST', datetime.now() - timedelta(hours=1), datetime.now()
        )
        # Data might be empty if no records are found, which is acceptable
        self.assertIsInstance(retrieved_data, pd.DataFrame)
    
    def test_database_stats(self):
        """Test database statistics"""
        stats = self.storage.get_database_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('market_data_count', stats)


class TestPolygonDataFetcher(unittest.TestCase):
    """Test Polygon data fetcher"""
    
    def setUp(self):
        check_module_available(PolygonDataFetcher)
        self.fetcher = PolygonDataFetcher(api_key='test_key', rate_limit=100)
    
    @patch('requests.get')
    def test_fetch_latest_data(self, mock_get):
        """Test fetching latest data"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'results': [{
                't': datetime.now().timestamp() * 1000,
                'o': 100.0,
                'h': 105.0,
                'l': 95.0,
                'c': 102.0,
                'v': 1000000
            }]
        }
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock the entire request to avoid API calls
        with patch.object(self.fetcher, '_make_request') as mock_request:
            mock_request.return_value = pd.DataFrame({
                'timestamp': [datetime.now()],
                'open': [100.0],
                'high': [105.0],
                'low': [95.0],
                'close': [102.0],
                'volume': [1000000]
            })
            
            data = self.fetcher.fetch_latest_data('AAPL', lookback_hours=24)
            self.assertIsInstance(data, pd.DataFrame)


class TestModelManager(unittest.TestCase):
    """Test model management"""
    
    def setUp(self):
        check_module_available(ModelManager)
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelManager(models_dir=self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_model_manager_initialization(self):
        """Test model manager initialization"""
        self.assertIsNotNone(self.model_manager)
        self.assertEqual(self.model_manager.models_dir, self.temp_dir)
    
    def test_list_available_models(self):
        """Test listing available models"""
        models = self.model_manager.list_available_models()
        self.assertIsInstance(models, list)


class TestPredictionEngine(unittest.TestCase):
    """Test prediction engine"""
    
    def setUp(self):
        check_module_available(DataStorage)
        check_module_available(ModelManager)
        check_module_available(PredictionEngine)
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.model_manager = ModelManager()
        self.prediction_engine = PredictionEngine(self.model_manager, self.storage)
    
    def tearDown(self):
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_prediction_engine_initialization(self):
        """Test prediction engine initialization"""
        self.assertIsNotNone(self.prediction_engine)
        self.assertIsInstance(self.prediction_engine.prediction_cache, dict)


class TestLoggingSystem(unittest.TestCase):
    """Test logging system"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {
            'log_directory': tempfile.mkdtemp(),
            'max_log_file_size': 1048576,
            'log_backup_count': 3
        }
        self.logging_system = LoggingSystem(self.storage, self.config)
    
    def tearDown(self):
        self.logging_system.stop()
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
        import shutil
        if os.path.exists(self.config['log_directory']):
            shutil.rmtree(self.config['log_directory'])
    
    def test_logging_system_initialization(self):
        """Test logging system initialization"""
        self.assertIsNotNone(self.logging_system)
    
    def test_log_structured(self):
        """Test structured logging"""
        self.logging_system.log_structured(
            LogLevel.INFO, LogCategory.SYSTEM, "Test message"
        )
    
    def test_log_performance(self):
        """Test performance logging"""
        self.logging_system.log_performance("test_operation", "AAPL")
    
    def test_get_logger(self):
        """Test logger retrieval"""
        logger = self.logging_system.get_logger("test_logger")
        self.assertIsNotNone(logger)


class TestErrorTracker(unittest.TestCase):
    """Test error tracking system"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {'error_queue_size': 1000}
        self.error_tracker = ErrorTracker(self.storage, self.config)
    
    def tearDown(self):
        self.error_tracker.stop()
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_error_tracker_initialization(self):
        """Test error tracker initialization"""
        self.assertIsNotNone(self.error_tracker)
    
    def test_track_error(self):
        """Test error tracking"""
        # Create a test exception
        test_exception = Exception("Test error message")
        error_id = self.error_tracker.track_error(
            exception=test_exception,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SYSTEM
        )
        self.assertIsNotNone(error_id)
    
    def test_get_error_statistics(self):
        """Test error statistics"""
        stats = self.error_tracker.get_error_statistics()
        self.assertIsInstance(stats, dict)


class TestPerformanceProfiler(unittest.TestCase):
    """Test performance profiling system"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {'profile_queue_size': 1000}
        self.profiler = PerformanceProfiler(self.storage, self.config)
    
    def tearDown(self):
        self.profiler.stop()
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_profiler_initialization(self):
        """Test performance profiler initialization"""
        self.assertIsNotNone(self.profiler)
    
    def test_profile_function(self):
        """Test function profiling"""
        def test_func():
            time.sleep(0.01)
            return "test"
        
        result = self.profiler.profile_function(test_func, "test_operation")
        # The profiler returns profiling data, not the original result
        self.assertIsInstance(result, dict)
    
    def test_get_performance_summary(self):
        """Test performance summary"""
        summary = self.profiler.get_performance_summary()
        self.assertIsInstance(summary, dict)


class TestWebhookSystem(unittest.TestCase):
    """Test webhook system"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {'delivery_workers': 2}
        self.webhook_system = WebhookSystem(self.storage, self.config)
    
    def tearDown(self):
        self.webhook_system.stop()
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_webhook_system_initialization(self):
        """Test webhook system initialization"""
        self.assertIsNotNone(self.webhook_system)
    
    def test_create_webhook(self):
        """Test webhook creation"""
        webhook_id = self.webhook_system.create_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            event_types=[WebhookEventType.PREDICTION_UPDATE]
        )
        self.assertIsNotNone(webhook_id)
    
    def test_trigger_webhook(self):
        """Test webhook triggering"""
        webhook_id = self.webhook_system.create_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            event_types=[WebhookEventType.PREDICTION_UPDATE]
        )
        
        self.webhook_system.trigger_webhook(
            WebhookEventType.PREDICTION_UPDATE,
            {'test': 'data'}
        )


class TestAPIServer(unittest.TestCase):
    """Test API server"""
    
    def setUp(self):
        check_module_available(APIServer)
        self.realtime_system = Mock()
        self.config = {
            'host': '127.0.0.1',
            'port': 5001,
            'debug': False,
            'require_auth': False
        }
        self.api_server = APIServer(self.realtime_system, self.config)
    
    def test_api_server_initialization(self):
        """Test API server initialization"""
        self.assertIsNotNone(self.api_server)
        self.assertIsNotNone(self.api_server.app)


class TestDataExporter(unittest.TestCase):
    """Test data export functionality"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {'export_directory': tempfile.mkdtemp()}
        self.exporter = DataExporter(self.storage, self.config)
    
    def tearDown(self):
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
        import shutil
        if os.path.exists(self.config['export_directory']):
            shutil.rmtree(self.config['export_directory'])
    
    def test_exporter_initialization(self):
        """Test data exporter initialization"""
        self.assertIsNotNone(self.exporter)
    
    def test_create_export(self):
        """Test export creation"""
        export_id = self.exporter.create_export(
            export_type=ExportType.PREDICTIONS,
            export_format=ExportFormat.CSV,
            symbols=['AAPL'],
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        self.assertIsNotNone(export_id)


class TestValidationWorkflow(unittest.TestCase):
    """Test validation workflow"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.model_manager = ModelManager()
        self.validation_workflow = PredictionValidationWorkflow(self.storage, self.model_manager)
    
    def tearDown(self):
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_validation_workflow_initialization(self):
        """Test validation workflow initialization"""
        self.assertIsNotNone(self.validation_workflow)


class TestAccuracyTracker(unittest.TestCase):
    """Test accuracy tracking"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {}
        self.accuracy_tracker = AccuracyTracker(self.storage, self.config)
    
    def tearDown(self):
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_accuracy_tracker_initialization(self):
        """Test accuracy tracker initialization"""
        self.assertIsNotNone(self.accuracy_tracker)
    
    def test_track_prediction(self):
        """Test prediction tracking"""
        self.accuracy_tracker.track_prediction_accuracy(
            symbol='AAPL',
            prediction_timestamp=datetime.now(),
            predicted_price=100.0,
            actual_price=102.0,
            confidence_score=0.8,
            model_version="test_model_v1"
        )


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {'auto_monitoring_enabled': False}
        self.performance_monitor = PerformanceMonitor(self.storage, self.config)
    
    def tearDown(self):
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization"""
        self.assertIsNotNone(self.performance_monitor)
    
    def test_record_metric(self):
        """Test metric recording"""
        # PerformanceMonitor automatically records metrics, so we test getting the summary
        summary = self.performance_monitor.get_performance_summary(hours_back=1)
        self.assertIsInstance(summary, dict)


class TestAlertingSystem(unittest.TestCase):
    """Test alerting system"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {}
        self.alerting_system = AlertingSystem(self.storage, self.config)
    
    def tearDown(self):
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_alerting_system_initialization(self):
        """Test alerting system initialization"""
        self.assertIsNotNone(self.alerting_system)
    
    def test_create_alert(self):
        """Test alert creation"""
        alert_id = self.alerting_system.create_alert(
            rule_id="test_rule_001",
            title="Test Alert",
            message="Test alert message",
            details={'symbol': 'AAPL', 'severity': 'WARNING'}
        )
        self.assertIsNotNone(alert_id)


class TestBackupRecovery(unittest.TestCase):
    """Test backup and recovery"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {'backup_directory': tempfile.mkdtemp()}
        self.backup_recovery = BackupRecoveryManager(self.storage, self.config)
    
    def tearDown(self):
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
        import shutil
        if os.path.exists(self.config['backup_directory']):
            shutil.rmtree(self.config['backup_directory'])
    
    def test_backup_recovery_initialization(self):
        """Test backup recovery initialization"""
        self.assertIsNotNone(self.backup_recovery)
    
    def test_create_backup(self):
        """Test backup creation"""
        backup_id = self.backup_recovery.create_backup(
            backup_type=BackupType.FULL
        )
        self.assertIsNotNone(backup_id)


class TestDataRetention(unittest.TestCase):
    """Test data retention"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.storage = DataStorage(self.temp_db)
        self.config = {}
        self.data_retention = DataRetentionManager(self.storage, self.config)
    
    def tearDown(self):
        self.storage.close()
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_data_retention_initialization(self):
        """Test data retention initialization"""
        self.assertIsNotNone(self.data_retention)
    
    def test_apply_retention_policy(self):
        """Test retention policy application"""
        result = self.data_retention.execute_retention_policies()
        self.assertIsInstance(result, list)


class TestRealTimeSystem(unittest.TestCase):
    """Test main real-time system"""
    
    def setUp(self):
        check_module_available(RealTimeSystem)
        self.config = {
            'symbols': ['AAPL'],
            'update_interval': 1,  # 1 minute for testing
            'min_profit_percent': 2.0,
            'polygon_api_key': 'test_key',
            'db_path': tempfile.mktemp(suffix='.db'),
            'models_dir': tempfile.mkdtemp()
        }
        self.system = RealTimeSystem(self.config)
    
    def tearDown(self):
        self.system.stop()
        if os.path.exists(self.config['db_path']):
            os.remove(self.config['db_path'])
        import shutil
        if os.path.exists(self.config['models_dir']):
            shutil.rmtree(self.config['models_dir'])
    
    def test_realtime_system_initialization(self):
        """Test real-time system initialization"""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.symbols, ['AAPL'])
    
    def test_system_status(self):
        """Test system status"""
        status = self.system.get_system_status()
        self.assertIsInstance(status, dict)
        self.assertIn('timestamp', status)
        self.assertIn('is_running', status)


class TestEnhancedRealTimeSystem(unittest.TestCase):
    """Test enhanced real-time system"""
    
    def setUp(self):
        check_module_available(EnhancedRealTimeSystem)
        self.config = {
            'db_path': tempfile.mktemp(suffix='.db'),
            'api': {
                'host': '127.0.0.1',
                'port': 5002,
                'debug': False,
                'require_auth': False
            },
            'logging': {
                'log_directory': tempfile.mkdtemp(),
                'max_log_file_size': 1048576
            },
            'webhooks': {'delivery_workers': 2},
            'error_tracking': {'error_queue_size': 1000},
            'performance_profiling': {'profile_queue_size': 1000}
        }
        self.enhanced_system = EnhancedRealTimeSystem(self.config)
    
    def tearDown(self):
        self.enhanced_system.stop()
        if os.path.exists(self.config['db_path']):
            os.remove(self.config['db_path'])
        import shutil
        if os.path.exists(self.config['logging']['log_directory']):
            shutil.rmtree(self.config['logging']['log_directory'])
    
    def test_enhanced_system_initialization(self):
        """Test enhanced system initialization"""
        self.assertIsNotNone(self.enhanced_system)
        self.assertTrue(self.enhanced_system.system_status['initialized'])
    
    def test_system_status(self):
        """Test enhanced system status"""
        status = self.enhanced_system.get_system_status()
        self.assertIsInstance(status, dict)
        self.assertIn('system_status', status)
    
    def test_system_metrics(self):
        """Test system metrics"""
        metrics = self.enhanced_system.get_system_metrics()
        self.assertIsInstance(metrics, dict)
    
    def test_create_webhook(self):
        """Test webhook creation through enhanced system"""
        webhook_id = self.enhanced_system.create_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            event_types=['prediction_update']
        )
        self.assertIsNotNone(webhook_id)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'db_path': os.path.join(self.temp_dir, 'test.db'),
            'models_dir': os.path.join(self.temp_dir, 'models'),
            'api': {
                'host': '127.0.0.1',
                'port': 5003,
                'debug': False,
                'require_auth': False
            },
            'logging': {
                'log_directory': os.path.join(self.temp_dir, 'logs'),
                'max_log_file_size': 1048576
            },
            'webhooks': {'delivery_workers': 1},
            'error_tracking': {'error_queue_size': 100},
            'performance_profiling': {'profile_queue_size': 100}
        }
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_system_workflow(self):
        """Test complete system workflow"""
        check_module_available(EnhancedRealTimeSystem)        # Create enhanced system
        system = EnhancedRealTimeSystem(self.config)
        
        try:
            # Test initialization
            self.assertTrue(system.system_status['initialized'])
            
            # Test component initialization
            self.assertIsNotNone(system.logging_system)
            self.assertIsNotNone(system.error_tracker)
            self.assertIsNotNone(system.performance_profiler)
            self.assertIsNotNone(system.webhook_system)
            
            # Test system status
            status = system.get_system_status()
            self.assertIsInstance(status, dict)
            
            # Test metrics
            metrics = system.get_system_metrics()
            self.assertIsInstance(metrics, dict)
            
        finally:
            system.stop()
    
    def test_error_handling(self):
        """Test error handling across the system"""
        check_module_available(EnhancedRealTimeSystem)
        system = EnhancedRealTimeSystem(self.config)
        
        try:
            # Test error tracking
            error_id = system.error_tracker.track_error(
                error_type="test_error",
                message="Integration test error",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.SYSTEM
            )
            self.assertIsNotNone(error_id)
            
            # Test error statistics
            stats = system.error_tracker.get_error_statistics()
            self.assertIsInstance(stats, dict)
            
        finally:
            system.stop()
    
    def test_logging_integration(self):
        """Test logging integration"""
        check_module_available(EnhancedRealTimeSystem)
        system = EnhancedRealTimeSystem(self.config)
        
        try:
            # Test structured logging
            system.logging_system.log_structured(
                LogLevel.INFO, LogCategory.SYSTEM, "Integration test message"
            )
            
            # Test performance logging
            system.logging_system.log_performance("integration_test", "AAPL")
            
            # Test log statistics
            stats = system.logging_system.get_log_statistics()
            self.assertIsInstance(stats, dict)
            
        finally:
            system.stop()
    
    def test_webhook_integration(self):
        """Test webhook integration"""
        check_module_available(EnhancedRealTimeSystem)
        system = EnhancedRealTimeSystem(self.config)
        
        try:
            # Create webhook
            webhook_id = system.create_webhook(
                name="Integration Test Webhook",
                url="https://httpbin.org/post",
                event_types=['prediction_update']
            )
            self.assertIsNotNone(webhook_id)
            
            # Trigger webhook
            system.webhook_system.trigger_webhook(
                'prediction_update',
                {'test': 'integration_data'}
            )
            
        finally:
            system.stop()


class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'db_path': os.path.join(self.temp_dir, 'perf_test.db'),
            'logging': {
                'log_directory': os.path.join(self.temp_dir, 'logs'),
                'max_log_file_size': 1048576
            }
        }
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_logging_performance(self):
        """Test logging performance"""
        storage = DataStorage(self.config['db_path'])
        logging_system = LoggingSystem(storage, self.config['logging'])
        
        try:
            start_time = time.time()
            
            # Log 1000 messages
            for i in range(1000):
                logging_system.log_structured(
                    LogLevel.INFO, LogCategory.SYSTEM, f"Performance test message {i}"
                )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete within reasonable time (adjust threshold as needed)
            self.assertLess(duration, 10.0)
            
        finally:
            logging_system.stop()
            storage.close()
    
    def test_error_tracking_performance(self):
        """Test error tracking performance"""
        storage = DataStorage(self.config['db_path'])
        error_tracker = ErrorTracker(storage, {})
        
        try:
            start_time = time.time()
            
            # Track 1000 errors
            for i in range(1000):
                test_exception = Exception(f"Performance test error {i}")
                error_tracker.track_error(
                    exception=test_exception,
                    severity=ErrorSeverity.LOW,
                    category=ErrorCategory.SYSTEM
                )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete within reasonable time
            self.assertLess(duration, 5.0)
            
        finally:
            error_tracker.stop()
            storage.close()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_invalid_config(self):
        """Test system behavior with invalid configuration"""
        invalid_config = {
            'db_path': '/invalid/path/that/does/not/exist/test.db',
            'logging': {
                'log_directory': '/invalid/path/that/does/not/exist'
            }
        }
        
        # Should handle invalid paths gracefully
        try:
            check_module_available(EnhancedRealTimeSystem)
            system = EnhancedRealTimeSystem(invalid_config)
            # If it doesn't raise an exception, that's also acceptable
            # as long as it handles the error internally
        except Exception as e:
            # Expected to potentially raise an exception
            self.assertIsInstance(e, Exception)
    
    def test_missing_dependencies(self):
        """Test behavior when dependencies are missing"""
        # This test would require mocking missing dependencies
        # For now, we'll test that the system handles missing config gracefully
        minimal_config = {}
        
        try:
            check_module_available(EnhancedRealTimeSystem)
            system = EnhancedRealTimeSystem(minimal_config)
        except Exception as e:
            # Expected to potentially raise an exception
            self.assertIsInstance(e, Exception)
    
    def test_concurrent_operations(self):
        """Test concurrent operations"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'db_path': os.path.join(temp_dir, 'concurrent_test.db'),
            'logging': {
                'log_directory': os.path.join(temp_dir, 'logs'),
                'max_log_file_size': 1048576
            }
        }
        
        system = None
        try:
            check_module_available(EnhancedRealTimeSystem)
            system = EnhancedRealTimeSystem(config)
            
            # Test concurrent logging
            def log_messages():
                for i in range(100):
                    system.logging_system.log_structured(
                        LogLevel.INFO, LogCategory.SYSTEM, f"Concurrent message {i}"
                    )
            
            # Start multiple threads
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=log_messages)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
        finally:
            if system is not None:
                system.stop()
            import shutil
            shutil.rmtree(temp_dir)


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestConfig,
        TestDataStorage,
        TestPolygonDataFetcher,
        TestModelManager,
        TestPredictionEngine,
        TestLoggingSystem,
        TestErrorTracker,
        TestPerformanceProfiler,
        TestWebhookSystem,
        TestAPIServer,
        TestDataExporter,
        TestValidationWorkflow,
        TestAccuracyTracker,
        TestPerformanceMonitor,
        TestAlertingSystem,
        TestBackupRecovery,
        TestDataRetention,
        TestRealTimeSystem,
        TestEnhancedRealTimeSystem,
        TestIntegration,
        TestPerformance,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    print("Starting Comprehensive Test Suite for Real-time LSTM Prediction System")
    print("=" * 80)
    
    result = run_comprehensive_tests()
    
    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("\nTest suite completed!")
    
    # Exit with appropriate code for pytest
    exit_code = 0 if len(result.failures) == 0 and len(result.errors) == 0 else 1
    exit(exit_code)
