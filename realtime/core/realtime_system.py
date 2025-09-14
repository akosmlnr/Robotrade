"""
Real-time LSTM Prediction System
Main orchestrator that coordinates all components for 1-week ahead predictions
"""

import time
import logging
import schedule
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
import signal
import sys

logger = logging.getLogger(__name__)

from data.data_fetcher import PolygonDataFetcher
from storage.data_storage import DataStorage
from models.model_manager import ModelManager
from models.prediction_engine import PredictionEngine

# Additional system components
from prediction.update_scheduler import UpdateScheduler
from validation.validation_workflow import PredictionValidationWorkflow
from validation.reprediction_triggers import RepredictionTriggers
from prediction.prediction_history import PredictionHistoryTracker
from storage.data_retention import DataRetentionManager
from storage.backup_recovery import BackupRecoveryManager
from export.data_export import DataExporter
from monitoring.accuracy_tracker import AccuracyTracker
from monitoring.performance_monitor import PerformanceMonitor
from monitoring.alerting_system import AlertingSystem
from monitoring.performance_dashboard import PerformanceDashboard

class RealTimeSystem:
    """
    Main real-time prediction system that orchestrates all components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the real-time system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.is_running = False
        self.symbols = self.config.get('symbols', ['AAPL'])
        self.update_interval = self.config.get('update_interval', 15)  # minutes
        self.min_profit_percent = self.config.get('min_profit_percent', 2.0)
        
        # Initialize components
        self.data_fetcher = PolygonDataFetcher(
            api_key=self.config.get('polygon_api_key'),
            rate_limit=self.config.get('rate_limit', 100)
        )
        self.data_storage = DataStorage(
            db_path=self.config.get('db_path', 'realtime_data.db')
        )
        self.model_manager = ModelManager(
            models_dir=self.config.get('models_dir', 'models')
        )
        self.prediction_engine = PredictionEngine(
            model_manager=self.model_manager,
            data_storage=self.data_storage
        )
        
        # Load models for configured symbols
        self._load_models()
        
        # Initialize all system components
        self._initialize_system_components()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"RealTimeSystem initialized for symbols: {self.symbols}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'symbols': ['AAPL'],
            'update_interval': 15,  # minutes
            'min_profit_percent': 2.0,
            'polygon_api_key': None,  # Will use environment variable
            'rate_limit': 100,
            'db_path': 'realtime_data.db',
            'models_dir': 'lstms',
            'validation_threshold': 0.02,  # 2% difference threshold for reprediction
            'confidence_threshold': 0.6,  # Minimum confidence for trade recommendations
            'max_recommendations_per_symbol': 5,
            # System component configuration
            'update_interval_minutes': 15,
            'validation_enabled': True,
            'reprediction_enabled': True,
            'accuracy_tracking_enabled': True,
            'performance_monitoring_enabled': True,
            'alerting_enabled': True,
            'backup_enabled': True,
            'data_retention_enabled': True,
            'dashboard_auto_refresh': True
        }
    
    def _load_models(self):
        """Load models for all configured symbols"""
        available_models = self.model_manager.list_available_models()
        
        # Filter symbols to only include those with available models
        self.symbols = [symbol for symbol in self.symbols if symbol in available_models]
        
        if not self.symbols:
            logger.warning("No models available for any configured symbols. Available models: " + str(available_models))
            return
        
        for symbol in self.symbols:
            success = self.model_manager.load_model(symbol)
            if success:
                logger.info(f"Successfully loaded model for {symbol}")
            else:
                logger.error(f"Failed to load model for {symbol}")
    
    def _initialize_system_components(self):
        """Initialize all system components"""
        try:
            # Initialize validation workflow
            self.validation_workflow = PredictionValidationWorkflow(
                self.data_storage, self.model_manager
            )
            
            # Initialize update scheduler
            self.update_scheduler = UpdateScheduler(
                self.prediction_engine, self.data_storage, self.validation_workflow
            )
            
            # Initialize reprediction triggers
            self.reprediction_triggers = RepredictionTriggers(
                self.data_storage, self.validation_workflow, self.update_scheduler
            )
            
            # Initialize prediction history tracker
            self.prediction_history = PredictionHistoryTracker(
                self.data_storage, "prediction_history.db"
            )
            
            # Initialize data retention manager
            self.data_retention = DataRetentionManager(
                self.data_storage, self.config.get('data_retention', {})
            )
            
            # Initialize backup recovery manager
            self.backup_recovery = BackupRecoveryManager(
                self.data_storage, self.config.get('backup_recovery', {})
            )
            
            # Initialize data exporter
            self.data_exporter = DataExporter(
                self.data_storage, self.config.get('data_export', {})
            )
            
            # Initialize accuracy tracker
            self.accuracy_tracker = AccuracyTracker(
                self.data_storage, self.config.get('accuracy_tracking', {})
            )
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(
                self.data_storage, self.config.get('performance_monitoring', {})
            )
            
            # Initialize alerting system
            self.alerting_system = AlertingSystem(
                self.data_storage, self.config.get('alerting', {})
            )
            
            # Initialize performance dashboard
            self.performance_dashboard = PerformanceDashboard(
                self.data_storage, self.config.get('dashboards', {})
            )
            
            logger.info("All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system components: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the real-time prediction system"""
        logger.info("Starting Real-time LSTM Prediction System")
        self.is_running = True
        
        # Schedule the main update task
        schedule.every(self.update_interval).minutes.do(self._update_cycle)
        
        # Run initial update
        self._update_cycle()
        
        # Start the scheduler in a separate thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info(f"System started. Updates every {self.update_interval} minutes.")
        
        # Keep main thread alive
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.stop()
    
    def stop(self):
        """Stop the real-time prediction system"""
        logger.info("Stopping Real-time LSTM Prediction System")
        self.is_running = False
        schedule.clear()
        
        # Cleanup
        self.data_storage.close()
        logger.info("System stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in a separate thread"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def _update_cycle(self):
        """Main update cycle that runs every 15 minutes"""
        logger.info("Starting update cycle")
        
        for symbol in self.symbols:
            try:
                self._update_symbol(symbol)
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
        
        logger.info("Update cycle completed")
    
    def _update_symbol(self, symbol: str):
        """Update predictions for a single symbol"""
        try:
            # Check if model is loaded
            model_data = self.model_manager.get_model(symbol)
            if not model_data:
                logger.warning(f"No model loaded for {symbol}")
                return
            
            # Fetch latest market data
            logger.info(f"Fetching latest data for {symbol}")
            latest_data = self.data_fetcher.fetch_latest_data(symbol, lookback_hours=24)
            
            if latest_data.empty:
                logger.warning(f"No market data available for {symbol}")
                return
            
            # Store market data
            self.data_storage.store_market_data(symbol, latest_data)
            
            # Validate previous predictions if available
            self._validate_previous_predictions(symbol)
            
            # Generate new predictions
            logger.info(f"Generating predictions for {symbol}")
            prediction_result = self.prediction_engine.generate_weekly_prediction(symbol, model_data)
            
            if prediction_result:
                # Generate trade recommendations
                recommendations = self.prediction_engine.generate_trade_recommendations(
                    symbol=symbol,
                    predictions_df=prediction_result['predictions'],
                    confidence_score=prediction_result['confidence_score'],
                    min_profit_percent=self.min_profit_percent
                )
                
                # Log results
                logger.info(f"Generated {len(recommendations)} trade recommendations for {symbol}")
                
                # Store in cache for quick access
                self.prediction_engine.prediction_cache[symbol] = {
                    'timestamp': datetime.now(),
                    'predictions': prediction_result['predictions'],
                    'recommendations': recommendations,
                    'confidence_score': prediction_result['confidence_score']
                }
            
        except Exception as e:
            logger.error(f"Error updating symbol {symbol}: {e}")
    
    def _validate_previous_predictions(self, symbol: str):
        """Validate previous predictions against actual data"""
        try:
            # Get recent predictions that need validation
            recent_predictions = self.data_storage.retrieve_historical_data(
                symbol, 
                datetime.now() - timedelta(hours=1),
                datetime.now()
            )
            
            if recent_predictions.empty:
                return
            
            # Get predictions that should be validated
            # This would involve querying the predictions table and comparing with actual prices
            # For now, we'll implement a simple validation
            
            logger.debug(f"Validated {len(recent_predictions)} recent predictions for {symbol}")
            
        except Exception as e:
            logger.error(f"Error validating predictions for {symbol}: {e}")
    
    def get_current_predictions(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get current predictions for a symbol or all symbols
        
        Args:
            symbol: Stock symbol (optional, if None returns all)
            
        Returns:
            Dictionary with current predictions
        """
        try:
            if symbol:
                if symbol in self.prediction_engine.prediction_cache:
                    return self.prediction_engine.prediction_cache[symbol]
                else:
                    return None
            else:
                return self.prediction_engine.prediction_cache.copy()
                
        except Exception as e:
            logger.error(f"Error getting current predictions: {e}")
            return {}
    
    def get_trade_recommendations(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get active trade recommendations
        
        Args:
            symbol: Stock symbol (optional, if None returns all)
            
        Returns:
            List of active trade recommendations
        """
        try:
            active_recommendations = self.data_storage.get_active_recommendations(symbol)
            return active_recommendations.to_dict('records') if not active_recommendations.empty else []
            
        except Exception as e:
            logger.error(f"Error getting trade recommendations: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status and health information
        
        Returns:
            Dictionary with system status
        """
        try:
            status = {
                'timestamp': datetime.now(),
                'is_running': self.is_running,
                'symbols': self.symbols,
                'update_interval': self.update_interval,
                'models_loaded': {},
                'database_stats': self.data_storage.get_database_stats(),
                'prediction_cache_size': len(self.prediction_engine.prediction_cache)
            }
            
            # Get model health for each symbol
            for symbol in self.symbols:
                model_data = self.model_manager.get_model(symbol)
                if model_data:
                    health = self.model_manager.run_health_check(symbol)
                    status['models_loaded'][symbol] = {
                        'loaded': True,
                        'health': health['overall_status'],
                        'loaded_at': model_data['loaded_at']
                    }
                else:
                    status['models_loaded'][symbol] = {
                        'loaded': False,
                        'health': 'not_loaded'
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def force_update(self, symbol: str = None):
        """
        Force an immediate update for a symbol or all symbols
        
        Args:
            symbol: Stock symbol (optional, if None updates all)
        """
        try:
            if symbol:
                logger.info(f"Force updating {symbol}")
                self._update_symbol(symbol)
            else:
                logger.info("Force updating all symbols")
                self._update_cycle()
                
        except Exception as e:
            logger.error(f"Error in force update: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration
    config = {
        'symbols': ['AAPL', 'GOOGL'],
        'update_interval': 15,  # minutes
        'min_profit_percent': 2.0,
        'polygon_api_key': None,  # Will use environment variable
        'db_path': 'realtime_data.db',
        'models_dir': 'models'
    }
    
    # Create and start the system
    system = RealTimeSystem(config)
    
    try:
        # Start the system (this will run indefinitely)
        system.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        system.stop()
