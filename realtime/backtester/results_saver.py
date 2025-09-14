"""
Text-based Results Saver for Backtester
Saves backtest results to text files instead of database
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TextResultsSaver:
    """
    Saves backtest results to text files in the backtester/results directory
    """
    
    def __init__(self, results_dir: str = "backtester/results"):
        """
        Initialize the text results saver
        
        Args:
            results_dir: Directory to save results (relative to project root)
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"TextResultsSaver initialized with directory: {results_dir}")
    
    def save_backtest_result(self, result: Dict[str, Any], symbol: str = None) -> str:
        """
        Save individual backtest result to text file
        
        Args:
            result: Backtest result dictionary
            symbol: Stock symbol (for filename)
            
        Returns:
            Path to saved file
        """
        try:
            if symbol is None:
                symbol = result.get('symbol', 'unknown')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_backtest_{timestamp}.txt"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write(f"BACKTEST RESULTS - {symbol.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Symbol: {symbol}\n")
                f.write("\n")
                
                # Basic info
                if 'historical_week' in result:
                    week = result['historical_week']
                    f.write("HISTORICAL WEEK:\n")
                    f.write(f"  Start Date: {week.get('start_date', 'N/A')}\n")
                    f.write(f"  End Date: {week.get('end_date', 'N/A')}\n")
                    f.write("\n")
                
                # Week statistics
                if 'week_statistics' in result:
                    stats = result['week_statistics']
                    f.write("WEEK STATISTICS:\n")
                    f.write(f"  Total Data Points: {stats.get('total_data_points', 'N/A')}\n")
                    f.write(f"  Duration (hours): {stats.get('duration_hours', 'N/A'):.1f}\n")
                    
                    if 'price_range' in stats:
                        price_range = stats['price_range']
                        f.write(f"  Price Range: ${price_range.get('min', 0):.2f} - ${price_range.get('max', 0):.2f}\n")
                        f.write(f"  Range: {price_range.get('range_percent', 0):.2f}%\n")
                    
                    if 'weekly_change' in stats:
                        change = stats['weekly_change']
                        f.write(f"  Weekly Change: {change.get('percent', 0):.2f}% (${change.get('absolute', 0):.2f})\n")
                    f.write("\n")
                
                # Simulation info
                f.write("SIMULATION INFO:\n")
                f.write(f"  Total Predictions: {result.get('total_predictions', 'N/A')}\n")
                f.write(f"  Total Duration: {result.get('total_duration_seconds', 'N/A'):.1f} seconds\n")
                f.write(f"  Data Updates: {result.get('total_data_updates', 'N/A')}\n")
                f.write("\n")
                
                # Accuracy metrics
                if 'accuracy_metrics' in result and result['accuracy_metrics']:
                    metrics = result['accuracy_metrics']
                    f.write("ACCURACY METRICS:\n")
                    f.write(f"  MAPE: {metrics.get('overall_mape', 0):.2f}%\n")
                    f.write(f"  MAE: ${metrics.get('overall_mae', 0):.2f}\n")
                    f.write(f"  RMSE: ${metrics.get('overall_rmse', 0):.2f}\n")
                    f.write(f"  Average Confidence: {metrics.get('average_confidence', 0):.3f}\n")
                    f.write(f"  Total Predictions Analyzed: {metrics.get('total_predictions', 0)}\n")
                    f.write("\n")
                
                # Error info
                if 'error' in result:
                    f.write("ERROR:\n")
                    f.write(f"  {result['error']}\n")
                    f.write("\n")
                
                # Detailed results
                if 'detailed_results' in result.get('accuracy_metrics', {}):
                    f.write("DETAILED RESULTS:\n")
                    for i, detail in enumerate(result['accuracy_metrics']['detailed_results'][:10]):  # Show first 10
                        f.write(f"  Prediction {i+1}:\n")
                        f.write(f"    Timestamp: {detail.get('prediction_timestamp', 'N/A')}\n")
                        f.write(f"    MAPE: {detail.get('mape', 0):.2f}%\n")
                        f.write(f"    MAE: ${detail.get('mae', 0):.2f}\n")
                        f.write(f"    RMSE: ${detail.get('rmse', 0):.2f}\n")
                        f.write(f"    Confidence: {detail.get('confidence_score', 0):.3f}\n")
                        f.write(f"    Data Points: {detail.get('data_points_used', 0)}\n")
                        f.write("\n")
                    
                    if len(result['accuracy_metrics']['detailed_results']) > 10:
                        f.write(f"  ... and {len(result['accuracy_metrics']['detailed_results']) - 10} more results\n")
                
                f.write("=" * 60 + "\n")
            
            logger.info(f"Backtest result saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving backtest result: {e}")
            return None
    
    def save_combined_results(self, results: Dict[str, Any]) -> str:
        """
        Save combined backtest results to text file
        
        Args:
            results: Combined results dictionary
            
        Returns:
            Path to saved file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"combined_backtest_{timestamp}.txt"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("COMBINED BACKTEST RESULTS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                
                # Summary
                f.write("SUMMARY:\n")
                f.write(f"  Total Runs: {results.get('total_runs', 0)}\n")
                f.write(f"  Successful Runs: {results.get('successful_runs', 0)}\n")
                f.write(f"  Failed Runs: {results.get('failed_runs', 0)}\n")
                f.write("\n")
                
                # Overall metrics
                if 'summary' in results and results['summary']:
                    summary = results['summary']
                    f.write("OVERALL ACCURACY METRICS:\n")
                    
                    if 'accuracy_metrics' in summary:
                        metrics = summary['accuracy_metrics']
                        
                        # MAPE
                        if 'mape' in metrics:
                            mape = metrics['mape']
                            f.write(f"  MAPE: {mape.get('mean', 0):.2f}% ± {mape.get('std', 0):.2f}%\n")
                            f.write(f"    Range: {mape.get('min', 0):.2f}% - {mape.get('max', 0):.2f}%\n")
                        
                        # MAE
                        if 'mae' in metrics:
                            mae = metrics['mae']
                            f.write(f"  MAE: ${mae.get('mean', 0):.2f} ± ${mae.get('std', 0):.2f}\n")
                            f.write(f"    Range: ${mae.get('min', 0):.2f} - ${mae.get('max', 0):.2f}\n")
                        
                        # RMSE
                        if 'rmse' in metrics:
                            rmse = metrics['rmse']
                            f.write(f"  RMSE: ${rmse.get('mean', 0):.2f} ± ${rmse.get('std', 0):.2f}\n")
                            f.write(f"    Range: ${rmse.get('min', 0):.2f} - ${rmse.get('max', 0):.2f}\n")
                        
                        # Confidence
                        if 'confidence' in metrics:
                            conf = metrics['confidence']
                            f.write(f"  Confidence: {conf.get('mean', 0):.3f} ± {conf.get('std', 0):.3f}\n")
                            f.write(f"    Range: {conf.get('min', 0):.3f} - {conf.get('max', 0):.3f}\n")
                    f.write("\n")
                
                # Symbol results
                if 'symbol_results' in results:
                    f.write("SYMBOL RESULTS:\n")
                    for symbol, symbol_results in results['symbol_results'].items():
                        f.write(f"  {symbol}:\n")
                        f.write(f"    Runs: {len(symbol_results)}\n")
                        
                        if symbol_results:
                            # Calculate symbol-specific metrics
                            symbol_mape = [r.get('accuracy_metrics', {}).get('overall_mape', 0) 
                                          for r in symbol_results if 'accuracy_metrics' in r]
                            symbol_confidence = [r.get('accuracy_metrics', {}).get('average_confidence', 0) 
                                               for r in symbol_results if 'accuracy_metrics' in r]
                            
                            if symbol_mape:
                                f.write(f"    Average MAPE: {sum(symbol_mape)/len(symbol_mape):.2f}%\n")
                            if symbol_confidence:
                                f.write(f"    Average Confidence: {sum(symbol_confidence)/len(symbol_confidence):.3f}\n")
                        f.write("\n")
                
                f.write("=" * 60 + "\n")
            
            logger.info(f"Combined backtest results saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving combined results: {e}")
            return None
    
    def get_results_directory(self) -> str:
        """Get the results directory path"""
        return self.results_dir
    
    def list_result_files(self) -> List[str]:
        """List all result files in the results directory"""
        try:
            if not os.path.exists(self.results_dir):
                return []
            
            files = [f for f in os.listdir(self.results_dir) if f.endswith('.txt')]
            return sorted(files)
            
        except Exception as e:
            logger.error(f"Error listing result files: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Test the results saver
    saver = TextResultsSaver()
    
    # Sample result
    sample_result = {
        'symbol': 'AAPL',
        'total_predictions': 672,
        'accuracy_metrics': {
            'overall_mape': 2.34,
            'overall_mae': 3.45,
            'overall_rmse': 4.56,
            'average_confidence': 0.78
        },
        'historical_week': {
            'start_date': '2022-06-13',
            'end_date': '2022-06-20'
        }
    }
    
    filepath = saver.save_backtest_result(sample_result)
    print(f"Sample result saved to: {filepath}")
    
    files = saver.list_result_files()
    print(f"Result files: {files}")
