"""
Example Usage of the Backtester
Demonstrates how to use the backtester for historical prediction simulation
"""

import logging
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path to import realtime modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import Backtester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def example_single_backtest():
    """Example of running a single backtest"""
    print("="*60)
    print("EXAMPLE 1: Single Symbol Backtest")
    print("="*60)
    
    # Create configuration
    config = {
        'symbols': ['AAPL'],
        'simulation_speed': 2.0,  # 2x speed for faster testing
        'models_dir': 'lstms',
        'output_dir': 'backtest_results',
        'verbose': True
    }
    
    # Create backtester
    backtester = Backtester(config)
    
    try:
        # Run backtest
        result = backtester.run_backtest('AAPL')
        
        if 'error' in result:
            print(f"Backtest failed: {result['error']}")
            return
        
        # Display results
        print(f"\nBacktest Results for {result['symbol']}:")
        print(f"- Historical Week: {result['historical_week']['start_date'].date()} to {result['historical_week']['end_date'].date()}")
        print(f"- Total Predictions: {result['total_predictions']}")
        print(f"- Simulation Duration: {result['total_duration_seconds']:.1f} seconds")
        
        if result.get('accuracy_metrics'):
            metrics = result['accuracy_metrics']
            print(f"- MAPE: {metrics.get('overall_mape', 0):.2f}%")
            print(f"- MAE: ${metrics.get('overall_mae', 0):.2f}")
            print(f"- RMSE: ${metrics.get('overall_rmse', 0):.2f}")
            print(f"- Average Confidence: {metrics.get('average_confidence', 0):.3f}")
        
        if result.get('week_statistics'):
            stats = result['week_statistics']
            print(f"- Data Points: {stats.get('total_data_points', 0)}")
            print(f"- Price Range: ${stats['price_range']['min']:.2f} - ${stats['price_range']['max']:.2f}")
            print(f"- Range: {stats['price_range']['range_percent']:.2f}%")
        
    except Exception as e:
        print(f"Error running single backtest: {e}")

def example_multiple_backtests():
    """Example of running multiple backtests"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Symbol Backtests")
    print("="*60)
    
    # Create configuration
    config = {
        'symbols': ['AAPL', 'GOOGL'],
        'simulation_speed': 3.0,  # 3x speed for faster testing
        'models_dir': 'lstms',
        'output_dir': 'backtest_results',
        'verbose': False
    }
    
    # Create backtester
    backtester = Backtester(config)
    
    try:
        # Run multiple backtests
        results = backtester.run_multiple_backtests(['AAPL', 'GOOGL'], num_runs_per_symbol=2)
        
        if 'error' in results:
            print(f"Multiple backtests failed: {results['error']}")
            return
        
        # Display summary
        print(f"\nMultiple Backtest Results:")
        print(f"- Total Runs: {results['total_runs']}")
        print(f"- Successful Runs: {results['successful_runs']}")
        print(f"- Failed Runs: {results['failed_runs']}")
        
        if results.get('summary') and results['summary'].get('accuracy_metrics'):
            metrics = results['summary']['accuracy_metrics']
            
            print(f"\nOverall Accuracy Metrics:")
            print(f"- MAPE: {metrics['mape']['mean']:.2f}% ± {metrics['mape']['std']:.2f}%")
            print(f"- MAE: ${metrics['mae']['mean']:.2f} ± ${metrics['mae']['std']:.2f}")
            print(f"- RMSE: ${metrics['rmse']['mean']:.2f} ± ${metrics['rmse']['std']:.2f}")
            print(f"- Confidence: {metrics['confidence']['mean']:.3f} ± {metrics['confidence']['std']:.3f}")
            
            print(f"\nAccuracy Ranges:")
            print(f"- MAPE: {metrics['mape']['min']:.2f}% - {metrics['mape']['max']:.2f}%")
            print(f"- MAE: ${metrics['mae']['min']:.2f} - ${metrics['mae']['max']:.2f}")
            print(f"- RMSE: ${metrics['rmse']['min']:.2f} - ${metrics['rmse']['max']:.2f}")
        
        # Display individual symbol results
        if results.get('symbol_results'):
            print(f"\nIndividual Symbol Results:")
            for symbol, symbol_results in results['symbol_results'].items():
                print(f"\n{symbol}:")
                print(f"  - Runs: {len(symbol_results)}")
                
                if symbol_results:
                    # Calculate symbol-specific metrics
                    symbol_mape = [r['accuracy_metrics']['overall_mape'] for r in symbol_results if 'accuracy_metrics' in r]
                    symbol_confidence = [r['accuracy_metrics']['average_confidence'] for r in symbol_results if 'accuracy_metrics' in r]
                    
                    if symbol_mape:
                        print(f"  - Average MAPE: {sum(symbol_mape)/len(symbol_mape):.2f}%")
                    if symbol_confidence:
                        print(f"  - Average Confidence: {sum(symbol_confidence)/len(symbol_confidence):.3f}")
        
    except Exception as e:
        print(f"Error running multiple backtests: {e}")

def example_custom_date_range():
    """Example of running backtest with custom date range"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Date Range Backtest")
    print("="*60)
    
    # Create configuration with custom date range
    config = {
        'symbols': ['AAPL'],
        'simulation_speed': 2.0,
        'models_dir': 'lstms',
        'output_dir': 'backtest_results',
        'min_date': datetime(2022, 1, 1),
        'max_date': datetime(2022, 12, 31),
        'verbose': True
    }
    
    # Create backtester
    backtester = Backtester(config)
    
    try:
        # Run backtest with custom date range
        result = backtester.run_backtest('AAPL')
        
        if 'error' in result:
            print(f"Custom date range backtest failed: {result['error']}")
            return
        
        # Display results
        print(f"\nCustom Date Range Backtest Results:")
        print(f"- Symbol: {result['symbol']}")
        print(f"- Historical Week: {result['historical_week']['start_date'].date()} to {result['historical_week']['end_date'].date()}")
        print(f"- Total Predictions: {result['total_predictions']}")
        
        if result.get('accuracy_metrics'):
            metrics = result['accuracy_metrics']
            print(f"- MAPE: {metrics.get('overall_mape', 0):.2f}%")
            print(f"- MAE: ${metrics.get('overall_mae', 0):.2f}")
            print(f"- RMSE: ${metrics.get('overall_rmse', 0):.2f}")
        
    except Exception as e:
        print(f"Error running custom date range backtest: {e}")

def example_with_callbacks():
    """Example of running backtest with custom callbacks"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Backtest with Custom Callbacks")
    print("="*60)
    
    # Custom callback functions
    def on_prediction(prediction_result):
        print(f"  → Prediction made at {prediction_result['prediction_timestamp']} "
              f"(confidence: {prediction_result['confidence_score']:.3f})")
    
    def on_update(data_update, prediction_result):
        if data_update['is_new_data']:
            print(f"  → New data at {data_update['timestamp']} "
                  f"({data_update['data_points']} points, price: ${data_update['latest_price']:.2f})")
    
    def on_completion(simulation_results):
        print(f"  → Simulation completed: {simulation_results['total_predictions']} predictions")
    
    # Create configuration
    config = {
        'symbols': ['AAPL'],
        'simulation_speed': 5.0,  # 5x speed for demo
        'models_dir': 'lstms',
        'output_dir': 'backtest_results',
        'verbose': False
    }
    
    # Create backtester
    backtester = Backtester(config)
    
    # Set custom callbacks
    backtester.prediction_simulator.set_callbacks(
        on_prediction=on_prediction,
        on_update=on_update,
        on_completion=on_completion
    )
    
    try:
        print("Running backtest with custom callbacks...")
        result = backtester.run_backtest('AAPL')
        
        if 'error' not in result:
            print(f"\nFinal Results:")
            print(f"- Total Predictions: {result['total_predictions']}")
            if result.get('accuracy_metrics'):
                metrics = result['accuracy_metrics']
                print(f"- MAPE: {metrics.get('overall_mape', 0):.2f}%")
        
    except Exception as e:
        print(f"Error running backtest with callbacks: {e}")

def main():
    """Main function to run all examples"""
    print("Backtester Examples")
    print("="*60)
    print("This script demonstrates various ways to use the backtester.")
    print("Make sure you have:")
    print("1. A valid Polygon.io API key in your environment")
    print("2. Trained LSTM models in the 'lstms' directory")
    print("3. Required dependencies installed")
    print()
    
    # Check if API key is available
    if not os.getenv('POLYGON_API_KEY'):
        print("WARNING: POLYGON_API_KEY environment variable not set!")
        print("Please set your Polygon.io API key before running examples.")
        return
    
    try:
        # Run examples
        example_single_backtest()
        example_multiple_backtests()
        example_custom_date_range()
        example_with_callbacks()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("Check the 'backtest_results' directory for saved results.")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")

if __name__ == "__main__":
    main()
