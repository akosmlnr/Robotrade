"""
Main Backtester Script
Orchestrates historical prediction simulation with incremental data updates
"""

import time
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import sys
import signal
import argparse

# Add parent directory to path to import realtime modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from historical_data_fetcher import HistoricalDataFetcher
from prediction_simulator import PredictionSimulator
from results_saver import TextResultsSaver
from models.model_manager import ModelManager
from storage.data_storage import DataStorage

logger = logging.getLogger(__name__)

class Backtester:
    """
    Main backtester that orchestrates historical prediction simulation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the backtester
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Validate required configuration
        self._validate_config()
        
        # Initialize components
        self.historical_fetcher = HistoricalDataFetcher(
            api_key=self.config.get('polygon_api_key'),
            rate_limit=self.config.get('rate_limit', 100)
        )
        
        self.data_storage = DataStorage(
            db_path=self.config.get('db_path', 'backtest_data.db')
        )
        
        self.model_manager = ModelManager(
            models_dir=self.config.get('models_dir', 'lstms')
        )
        
        # Load models for configured symbols
        self._load_models()
        
        self.prediction_simulator = PredictionSimulator(
            model_manager=self.model_manager,
            data_storage=self.data_storage,
            simulation_speed=self.config.get('simulation_speed', 1.0)
        )
        
        # Initialize text results saver
        self.results_saver = TextResultsSaver(
            results_dir=self.config.get('results_dir', 'backtester/results')
        )
        
        # Simulation state
        self.is_running = False
        self.current_results = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Set up callbacks
        self._setup_callbacks()
        
        logger.info("Backtester initialized")
    
    def _load_models(self):
        """Load models for all configured symbols"""
        available_models = self.model_manager.list_available_models()
        
        # Filter symbols to only include those with available models
        configured_symbols = self.config.get('symbols', ['AAPL'])
        self.config['symbols'] = [symbol for symbol in configured_symbols if symbol in available_models]
        
        if not self.config['symbols']:
            logger.warning("No models available for any configured symbols. Available models: " + str(available_models))
            return
        
        for symbol in self.config['symbols']:
            success = self.model_manager.load_model(symbol)
            if success:
                logger.info(f"Successfully loaded model for {symbol}")
            else:
                logger.error(f"Failed to load model for {symbol}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        try:
            # Check if models directory exists
            models_dir = self.config.get('models_dir', 'lstms')
            if not os.path.exists(models_dir):
                logger.warning(f"Models directory does not exist: {models_dir}")
            
            # Check if symbols are provided
            symbols = self.config.get('symbols', [])
            if not symbols:
                raise ValueError("No symbols configured for backtesting")
            
            # Validate simulation speed
            speed = self.config.get('simulation_speed', 1.0)
            if speed <= 0:
                raise ValueError("Simulation speed must be positive")
            
            logger.info("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'symbols': ['AAPL'],
            'simulation_speed': 1.0,  # 1.0 = real-time, 2.0 = 2x speed
            'polygon_api_key': None,  # Will use environment variable
            'rate_limit': 100,
            'db_path': 'backtest_data.db',
            'models_dir': 'lstms',
            'min_date': None,  # Will default to 2 years ago
            'max_date': None,  # Will default to 1 year ago
            'output_dir': 'backtest_results',
            'save_results': True,
            'verbose': False
        }
    
    def _setup_callbacks(self):
        """Set up callbacks for the prediction simulator"""
        def on_prediction(prediction_result):
            """Callback for when a prediction is made"""
            if prediction_result['prediction_horizon'] == '1 week (daily)':
                # Weekly prediction format
                logger.info(f"Weekly prediction made for {prediction_result['symbol']} at "
                           f"{prediction_result['prediction_timestamp']} "
                           f"(confidence: {prediction_result['confidence_score']:.3f})")
                logger.info(f"  → Predicted week-end price: ${prediction_result['predicted_week_end_price']:.2f} "
                           f"from ${prediction_result['latest_actual_price']:.2f}")
            else:
                # Legacy format
                logger.info(f"Prediction made for {prediction_result['symbol']} at "
                           f"{prediction_result['prediction_timestamp']} "
                           f"(confidence: {prediction_result['confidence_score']:.3f})")
        
        def on_update(data_update, prediction_result):
            """Callback for when data is updated"""
            logger.debug(f"Data update for {data_update['symbol']} at {data_update['timestamp']} "
                        f"({data_update['data_points']} data points)")
        
        def on_completion(simulation_results):
            """Callback for when simulation completes"""
            logger.info(f"Simulation completed for {simulation_results['symbol']}")
            logger.info(f"Total predictions: {simulation_results['total_predictions']}")
            
            if simulation_results.get('accuracy_metrics'):
                metrics = simulation_results['accuracy_metrics']
                logger.info(f"Accuracy - MAPE: {metrics.get('overall_mape', 0):.2f}%, "
                           f"MAE: ${metrics.get('overall_mae', 0):.2f}, "
                           f"RMSE: ${metrics.get('overall_rmse', 0):.2f}")
            
            self.current_results = simulation_results
            
            # Save results if configured
            if self.config.get('save_results', True):
                self._save_results(simulation_results)
        
        self.prediction_simulator.set_callbacks(
            on_prediction=on_prediction,
            on_update=on_update,
            on_completion=on_completion
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def run_backtest(self, symbol: str = None, 
                    min_date: datetime = None, 
                    max_date: datetime = None) -> Dict[str, Any]:
        """
        Run a backtest for a specific symbol
        
        Args:
            symbol: Stock symbol to backtest (default: first symbol in config)
            min_date: Minimum date for historical data (default: from config)
            max_date: Maximum date for historical data (default: from config)
            
        Returns:
            Backtest results dictionary
        """
        try:
            # Use provided symbol or default from config
            if symbol is None:
                symbol = self.config['symbols'][0]
            
            # Use provided dates or defaults from config
            if min_date is None:
                min_date = self.config.get('min_date')
            if max_date is None:
                max_date = self.config.get('max_date')
            
            logger.info(f"Starting backtest for {symbol}")
            
            # Check if model is available
            if not self.model_manager.get_model(symbol):
                logger.error(f"No model available for {symbol}")
                return {'error': f'No model available for {symbol}'}
            
            # Select random historical week
            start_date, end_date = self.historical_fetcher.get_random_historical_week(
                symbol, min_date, max_date
            )
            
            logger.info(f"Selected historical week: {start_date.date()} to {end_date.date()}")
            
            # Fetch historical data for the week
            historical_data = self.historical_fetcher.fetch_historical_week_data(
                symbol, start_date, end_date
            )
            
            if historical_data.empty:
                logger.error(f"No historical data found for {symbol}")
                return {'error': f'No historical data found for {symbol}'}
            
            # Calculate week statistics
            week_stats = self.historical_fetcher.calculate_week_statistics(historical_data)
            logger.info(f"Week statistics: {week_stats}")
            
            # Create incremental data updates
            data_updates = self.historical_fetcher.simulate_incremental_data_updates(
                symbol, historical_data, start_date
            )
            
            if not data_updates:
                logger.error(f"Failed to create data updates for {symbol}")
                return {'error': f'Failed to create data updates for {symbol}'}
            
            logger.info(f"Created {len(data_updates)} data updates for simulation")
            
            # Pass full historical data to prediction simulator for validation
            self.prediction_simulator.full_historical_data = historical_data
            
            # Start prediction simulation
            success = self.prediction_simulator.start_simulation(symbol, data_updates)
            
            if not success:
                logger.error(f"Failed to start prediction simulation for {symbol}")
                return {'error': f'Failed to start prediction simulation for {symbol}'}
            
            # Wait for simulation to complete
            self.is_running = True
            logger.info("Simulation started, waiting for completion...")
            
            # Wait for simulation to complete - it should be very fast now
            while self.prediction_simulator.is_running:
                time.sleep(0.05)  # Check frequently since it runs very fast
                
                # Print progress every 25 updates since each generates 672 predictions
                status = self.prediction_simulator.get_simulation_status()
                if status.get('progress_percent', 0) > 0:
                    current_index = status.get('current_update_index', 0)
                    if current_index % 25 == 0 and current_index > 0:
                        total_predictions = current_index * 672  # Each update generates 672 predictions
                        logger.info(f"Progress: {status['progress_percent']:.1f}% "
                                   f"({status['current_update_index']}/{status['total_updates']} updates, "
                                   f"{total_predictions} predictions)")
            
            # Get final results
            results = self.prediction_simulator._get_simulation_results()
            results['symbol'] = symbol
            results['historical_week'] = {
                'start_date': start_date,
                'end_date': end_date
            }
            results['week_statistics'] = week_stats
            
            logger.info(f"Backtest completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")
            return {'error': str(e)}
    
    def run_multiple_backtests(self, symbols: List[str] = None, 
                              num_runs_per_symbol: int = 3) -> Dict[str, Any]:
        """
        Run multiple backtests for different symbols and/or multiple runs
        
        Args:
            symbols: List of symbols to backtest (default: from config)
            num_runs_per_symbol: Number of runs per symbol
            
        Returns:
            Combined results dictionary
        """
        try:
            if symbols is None:
                symbols = self.config['symbols']
            
            logger.info(f"Starting multiple backtests: {symbols}, {num_runs_per_symbol} runs each")
            
            all_results = []
            symbol_results = {}
            
            for symbol in symbols:
                logger.info(f"Running backtests for {symbol}")
                symbol_results[symbol] = []
                
                for run_num in range(num_runs_per_symbol):
                    logger.info(f"Run {run_num + 1}/{num_runs_per_symbol} for {symbol}")
                    
                    result = self.run_backtest(symbol)
                    
                    if 'error' not in result:
                        result['run_number'] = run_num + 1
                        symbol_results[symbol].append(result)
                        all_results.append(result)
                    else:
                        logger.error(f"Run {run_num + 1} failed for {symbol}: {result['error']}")
            
            # Calculate summary statistics
            summary = self._calculate_summary_statistics(all_results)
            
            combined_results = {
                'summary': summary,
                'individual_results': all_results,
                'symbol_results': symbol_results,
                'total_runs': len(all_results),
                'successful_runs': len([r for r in all_results if 'error' not in r]),
                'failed_runs': len([r for r in all_results if 'error' in r])
            }
            
            # Save combined results
            if self.config.get('save_results', True):
                self._save_combined_results(combined_results)
            
            logger.info(f"Multiple backtests completed: {combined_results['successful_runs']}/{combined_results['total_runs']} successful")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error running multiple backtests: {e}")
            return {'error': str(e)}
    
    def _calculate_summary_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from multiple backtest results"""
        try:
            if not results:
                return {}
            
            # Extract accuracy metrics
            all_mape = []
            all_mae = []
            all_rmse = []
            all_confidence = []
            
            for result in results:
                if 'accuracy_metrics' in result and result['accuracy_metrics']:
                    metrics = result['accuracy_metrics']
                    all_mape.append(metrics.get('overall_mape', 0))
                    all_mae.append(metrics.get('overall_mae', 0))
                    all_rmse.append(metrics.get('overall_rmse', 0))
                    all_confidence.append(metrics.get('average_confidence', 0))
            
            summary = {
                'total_runs': len(results),
                'accuracy_metrics': {
                    'mape': {
                        'mean': np.mean(all_mape) if all_mape else 0,
                        'std': np.std(all_mape) if all_mape else 0,
                        'min': np.min(all_mape) if all_mape else 0,
                        'max': np.max(all_mape) if all_mape else 0
                    },
                    'mae': {
                        'mean': np.mean(all_mae) if all_mae else 0,
                        'std': np.std(all_mae) if all_mae else 0,
                        'min': np.min(all_mae) if all_mae else 0,
                        'max': np.max(all_mae) if all_mae else 0
                    },
                    'rmse': {
                        'mean': np.mean(all_rmse) if all_rmse else 0,
                        'std': np.std(all_rmse) if all_rmse else 0,
                        'min': np.min(all_rmse) if all_rmse else 0,
                        'max': np.max(all_rmse) if all_rmse else 0
                    },
                    'confidence': {
                        'mean': np.mean(all_confidence) if all_confidence else 0,
                        'std': np.std(all_confidence) if all_confidence else 0,
                        'min': np.min(all_confidence) if all_confidence else 0,
                        'max': np.max(all_confidence) if all_confidence else 0
                    }
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
            return {}
    
    def _save_results(self, results: Dict[str, Any]):
        """Save individual backtest results to text file"""
        try:
            symbol = results.get('symbol', 'unknown')
            filepath = self.results_saver.save_backtest_result(results, symbol)
            
            if filepath:
                logger.info(f"Results saved to {filepath}")
            else:
                logger.error("Failed to save results")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _save_combined_results(self, results: Dict[str, Any]):
        """Save combined backtest results to text file"""
        try:
            filepath = self.results_saver.save_combined_results(results)
            
            if filepath:
                logger.info(f"Combined results saved to {filepath}")
            else:
                logger.error("Failed to save combined results")
            
        except Exception as e:
            logger.error(f"Error saving combined results: {e}")
    
    def stop(self):
        """Stop the backtester"""
        try:
            logger.info("Stopping backtester")
            self.prediction_simulator.stop_simulation()
            self.is_running = False
            logger.info("Backtester stopped")
            
        except Exception as e:
            logger.error(f"Error stopping backtester: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current backtester status"""
        try:
            status = {
                'is_running': self.is_running,
                'current_results': self.current_results is not None,
                'simulation_status': self.prediction_simulator.get_simulation_status(),
                'cache_info': self.historical_fetcher.get_cache_info()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Run LSTM prediction backtests')
    parser.add_argument('--symbol', type=str, help='Stock symbol to backtest')
    parser.add_argument('--symbols', nargs='+', help='Multiple symbols to backtest')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per symbol')
    parser.add_argument('--speed', type=float, default=1.0, help='Simulation speed multiplier')
    parser.add_argument('--models-dir', type=str, default='lstms', help='Models directory')
    parser.add_argument('--output-dir', type=str, default='backtest_results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--min-date', type=str, help='Minimum date (YYYY-MM-DD)')
    parser.add_argument('--max-date', type=str, help='Maximum date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse dates
    min_date = None
    max_date = None
    if args.min_date:
        min_date = datetime.strptime(args.min_date, '%Y-%m-%d')
    if args.max_date:
        max_date = datetime.strptime(args.max_date, '%Y-%m-%d')
    
    # Create configuration
    config = {
        'symbols': args.symbols or [args.symbol] or ['AAPL'],
        'simulation_speed': args.speed,
        'models_dir': args.models_dir,
        'output_dir': args.output_dir,
        'min_date': min_date,
        'max_date': max_date,
        'verbose': args.verbose
    }
    
    # Create and run backtester
    backtester = Backtester(config)
    
    try:
        if len(config['symbols']) == 1:
            # Single symbol backtest
            result = backtester.run_backtest(config['symbols'][0])
        else:
            # Multiple symbol backtest
            result = backtester.run_multiple_backtests(config['symbols'], args.runs)
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(json.dumps(result, indent=2, default=str))
        
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
        backtester.stop()
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
