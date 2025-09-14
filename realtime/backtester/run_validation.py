#!/usr/bin/env python3
"""
Run backtester with validation
This script runs the backtester and then validates the predictions against actual data
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime

# Add the current directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules from the current directory
from backtester import Backtester
from validation import BacktestValidator

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_backtest_with_validation(symbol: str = "AAPL"):
    """
    Run backtest and validate results
    
    Args:
        symbol: Stock symbol to backtest
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting backtest with validation for {symbol}")
        
        # Initialize backtester
        backtester = Backtester()
        
        # Run backtest
        logger.info("Running backtest...")
        backtest_results = backtester.run_backtest(symbol)
        
        if 'error' in backtest_results:
            logger.error(f"Backtest failed: {backtest_results['error']}")
            return backtest_results
        
        # Get actual data from simulation
        actual_data = backtester.prediction_simulator.get_actual_data_for_validation()
        
        if actual_data.empty:
            logger.warning("No actual data available for validation")
            return {"error": "No actual data available for validation"}
        
        logger.info(f"Actual data for validation: {len(actual_data)} records")
        logger.info(f"Actual data range: {actual_data.index.min()} to {actual_data.index.max()}")
        logger.info(f"Actual data columns: {actual_data.columns.tolist()}")
        
        # Initialize validator with historical data for context
        validator = BacktestValidator(actual_data_for_validation=actual_data)
        
        # Validate predictions
        logger.info("Validating predictions...")
        prediction_results = backtest_results.get('prediction_history', [])
        
        validation_results = validator.validate_predictions(prediction_results, actual_data)
        
        if 'error' in validation_results:
            logger.error(f"Validation failed: {validation_results['error']}")
            return validation_results
        
        # Save validation results - DISABLED to prevent JSON file creation
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # results_file = validator.save_validation_results(validation_results, f"validation_{symbol}_{timestamp}.json")
        results_file = None
        
        # Combine results
        combined_results = {
            "backtest_results": backtest_results,
            "validation_results": validation_results,
            "validation_file": results_file
        }
        
        logger.info(f"Backtest and validation completed successfully!")
        logger.info(f"Validation results saved to: {results_file}")
        
        # Print summary
        if 'accuracy_metrics' in validation_results:
            metrics = validation_results['accuracy_metrics']
            if 'overall_accuracy' in metrics:
                overall = metrics['overall_accuracy']
                logger.info(f"Accuracy Summary:")
                logger.info(f"  MAPE: {overall.get('mape', 'N/A'):.2f}%")
                logger.info(f"  MAE: ${overall.get('mae', 'N/A'):.2f}")
                logger.info(f"  RMSE: ${overall.get('rmse', 'N/A'):.2f}")
                logger.info(f"  Directional Accuracy: {overall.get('directional_accuracy', 'N/A'):.1f}%")
                logger.info(f"  Correlation: {overall.get('correlation', 'N/A'):.3f}")
        
        return combined_results
        
    except Exception as e:
        logger.error(f"Error running backtest with validation: {e}")
        return {"error": str(e)}

def main():
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run backtester with validation')
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol to backtest')
    parser.add_argument('--speed', type=float, default=1.0, help='Simulation speed multiplier')
    parser.add_argument('--runs', type=int, default=1, help='Number of backtest runs')
    
    args = parser.parse_args()
    
    # Run backtest with validation
    results = run_backtest_with_validation(symbol=args.symbol)
    
    if 'error' in results:
        logger.error(f"Failed: {results['error']}")
        sys.exit(1)
    else:
        logger.info("Success!")
        sys.exit(0)

if __name__ == "__main__":
    main()
