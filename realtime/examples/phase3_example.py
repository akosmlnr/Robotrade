"""
Phase 3: Real-Time Processing Example
Demonstrates how to use the complete Phase 3 system for real-time LSTM prediction management
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import components
from storage.data_storage import DataStorage
from models.model_manager import ModelManager
from models.prediction_engine import PredictionEngine
from core.realtime_system import RealTimeSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_system() -> RealTimeSystem:
    """Create and configure Real-Time System"""
    
    # Initialize core components
    data_storage = DataStorage("phase3_example.db")
    model_manager = ModelManager()
    prediction_engine = PredictionEngine(data_storage, model_manager)
    
    # Phase 3 configuration
    phase3_config = {
        'enabled_components': [
            'update_scheduler',
            'validation_workflow', 
            'reprediction_triggers',
            'prediction_history',
            'data_retention',
            'backup_recovery',
            'data_export',
            'accuracy_tracking',
            'performance_monitoring',
            'alerting',
            'dashboards'
        ],
        'update_interval_minutes': 15,
        'validation_enabled': True,
        'reprediction_enabled': True,
        'accuracy_tracking_enabled': True,
        'performance_monitoring_enabled': True,
        'alerting_enabled': True,
        'backup_enabled': True,
        'data_retention_enabled': True,
        'dashboard_auto_refresh': True,
        'symbols_to_monitor': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        
        # Component-specific configurations
        'data_retention': {
            'retention_policies': {
                'raw_data': {'days': 90, 'enabled': True},
                'predictions': {'days': 30, 'enabled': True},
                'validation_results': {'days': 60, 'enabled': True},
                'performance_metrics': {'days': 180, 'enabled': True},
                'system_alerts': {'days': 30, 'enabled': True}
            },
            'cleanup_schedule': 'daily',
            'auto_cleanup_enabled': True
        },
        
        'backup_recovery': {
            'backup_schedule': 'daily',
            'backup_retention_days': 30,
            'auto_backup_enabled': True,
            'backup_location': './backups/',
            'compression_enabled': True
        },
        
        'data_export': {
            'export_formats': ['csv', 'json', 'excel'],
            'export_location': './exports/',
            'auto_cleanup_exports': True,
            'export_retention_days': 7
        },
        
        'accuracy_tracking': {
            'tracking_enabled': True,
            'accuracy_threshold': 0.7,
            'confidence_threshold': 0.8,
            'tracking_window_days': 30
        },
        
        'performance_monitoring': {
            'monitoring_enabled': True,
            'auto_monitoring_enabled': True,
            'monitoring_interval_minutes': 5,
            'metrics_to_track': [
                'prediction_accuracy',
                'model_confidence',
                'validation_scores',
                'system_performance',
                'data_quality'
            ],
            'thresholds': {
                'prediction_accuracy': {'warning': 0.6, 'critical': 0.4},
                'model_confidence': {'warning': 0.7, 'critical': 0.5},
                'validation_scores': {'warning': 0.6, 'critical': 0.4},
                'system_performance': {'warning': 0.8, 'critical': 0.6},
                'data_quality': {'warning': 0.7, 'critical': 0.5}
            }
        },
        
        'alerting': {
            'alerting_enabled': True,
            'alert_rules': {
                'prediction_failure': {
                    'enabled': True,
                    'severity': 'high',
                    'cooldown_minutes': 15
                },
                'accuracy_drop': {
                    'enabled': True,
                    'severity': 'medium',
                    'threshold': 0.6,
                    'cooldown_minutes': 30
                },
                'system_error': {
                    'enabled': True,
                    'severity': 'high',
                    'cooldown_minutes': 5
                },
                'data_quality_issue': {
                    'enabled': True,
                    'severity': 'medium',
                    'threshold': 0.7,
                    'cooldown_minutes': 60
                }
            },
            'notification_channels': {
                'console': {'enabled': True},
                'file': {'enabled': True, 'file_path': './alerts.log'},
                'email': {'enabled': False},
                'webhook': {'enabled': False}
            }
        },
        
        'dashboards': {
            'dashboard_auto_refresh': True,
            'refresh_interval_minutes': 60,
            'default_dashboard_type': 'system_overview',
            'dashboard_location': './dashboards/',
            'auto_generate_dashboards': True
        }
    }
    
    # Create system with all components
    system = RealTimeSystem(config=phase3_config)
    
    return system

def demonstrate_system_features(system: RealTimeSystem):
    """Demonstrate key system features"""
    
    logger.info("=== Real-Time System Demonstration ===")
    
    # 1. Start the system
    logger.info("1. Starting Real-Time System...")
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    success = system.start_system(symbols)
    
    if not success:
        logger.error("Failed to start system")
        return
    
    logger.info("System started successfully")
    
    # 2. Create dashboards
    logger.info("2. Creating performance dashboards...")
    
    # System overview dashboard
    dashboard_id = system.create_dashboard('system_overview')
    logger.info(f"Created system overview dashboard: {dashboard_id}")
    
    # Performance metrics dashboard
    dashboard_id = system.create_dashboard('performance_metrics')
    logger.info(f"Created performance metrics dashboard: {dashboard_id}")
    
    # 3. Export data
    logger.info("3. Demonstrating data export...")
    
    export_id = system.export_data(
        export_type='predictions',
        symbols=symbols,
        days_back=7,
        format='csv'
    )
    logger.info(f"Created data export: {export_id}")
    
    # 4. Get system status
    logger.info("4. Getting system status...")
    status = system.get_system_status()
    logger.info(f"System running: {status['system_running']}")
    logger.info(f"Uptime: {status['uptime_seconds']:.2f} seconds")
    
    # 5. Get performance summary
    logger.info("5. Getting performance summary...")
    summary = system.get_performance_summary()
    logger.info("Performance summary retrieved successfully")
    
    # 6. Let the system run for a while
    logger.info("6. Letting system run for 2 minutes...")
    time.sleep(120)  # Run for 2 minutes
    
    # 7. Stop the system
    logger.info("7. Stopping Phase 3 system...")
    success = system.stop_system()
    
    if success:
        logger.info("Phase 3 system stopped successfully")
    else:
        logger.error("Failed to stop Phase 3 system")

def demonstrate_individual_components():
    """Demonstrate individual Phase 3 components"""
    
    logger.info("=== Individual Component Demonstration ===")
    
    # Initialize data storage
    data_storage = DataStorage("component_demo.db")
    
    # 1. Validation Workflow
    logger.info("1. Testing Validation Workflow...")
    from ..validation.validation_workflow import PredictionValidationWorkflow
    from ..models.model_manager import ModelManager
    
    model_manager = ModelManager()
    validation_workflow = PredictionValidationWorkflow(data_storage, model_manager)
    
    # Test validation
    test_prediction = {
        'symbol': 'AAPL',
        'predictions': [150.0, 151.0, 152.0, 153.0, 154.0],
        'confidence': 0.85,
        'timestamp': datetime.now()
    }
    
    validation_result = validation_workflow.validate_prediction(test_prediction)
    logger.info(f"Validation result: {validation_result}")
    
    # 2. Reprediction Triggers
    logger.info("2. Testing Reprediction Triggers...")
    from ..validation.reprediction_triggers import RepredictionTriggers, TriggerType
    
    reprediction_triggers = RepredictionTriggers(data_storage, validation_workflow, None)
    
    # Test trigger creation
    trigger = reprediction_triggers.create_trigger(
        symbol='AAPL',
        trigger_type=TriggerType.VALIDATION_FAILURE,
        threshold=0.5,
        enabled=True
    )
    logger.info(f"Created trigger: {trigger}")
    
    # 3. Data Export
    logger.info("3. Testing Data Export...")
    from ..export.data_export import DataExporter, ExportType, ExportFormat
    
    data_exporter = DataExporter(data_storage, {})
    
    export_id = data_exporter.create_export(
        export_type=ExportType.PREDICTIONS,
        export_format=ExportFormat.CSV,
        symbols=['AAPL'],
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now()
    )
    logger.info(f"Created export: {export_id}")
    
    # 4. Performance Monitoring
    logger.info("4. Testing Performance Monitoring...")
    from ..monitoring.performance_monitor import PerformanceMonitor
    
    performance_monitor = PerformanceMonitor(data_storage, {})
    
    # Test metric collection
    performance_monitor.collect_metric('prediction_accuracy', 0.75, {'symbol': 'AAPL'})
    performance_monitor.collect_metric('model_confidence', 0.82, {'symbol': 'AAPL'})
    
    metrics = performance_monitor.get_metrics('prediction_accuracy', days_back=7)
    logger.info(f"Collected {len(metrics)} accuracy metrics")
    
    # 5. Alerting System
    logger.info("5. Testing Alerting System...")
    from ..monitoring.alerting_system import AlertingSystem
    
    alerting_system = AlertingSystem(data_storage, {})
    
    # Test alert creation
    alert_id = alerting_system.create_alert(
        rule_id='test_alert',
        title='Test Alert',
        message='This is a test alert',
        details={'test': True}
    )
    logger.info(f"Created alert: {alert_id}")
    
    logger.info("Individual component demonstration completed")

def main():
    """Main demonstration function"""
    
    try:
        logger.info("Starting Real-Time Processing Demonstration")
        
        # Create system
        system = create_system()
        
        # Demonstrate integrated system
        demonstrate_system_features(system)
        
        # Demonstrate individual components
        demonstrate_individual_components()
        
        logger.info("System demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in system demonstration: {e}")
        raise

if __name__ == "__main__":
    main()
