"""
Backtester Validation System
Compares predicted values to actual values and calculates accuracy metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import os
from io import StringIO

logger = logging.getLogger(__name__)

class BacktestValidator:
    """
    Validates backtest predictions against actual market data
    """
    
    def __init__(self, results_dir: str = "backtester/results", actual_data_for_validation: pd.DataFrame = None):
        """
        Initialize the validator
        
        Args:
            results_dir: Directory containing backtest results
            actual_data_for_validation: Historical data for context in plots
        """
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "plots")
        self.actual_data_for_validation = actual_data_for_validation
        
        # Create plots directory if it doesn't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        
        logger.info(f"BacktestValidator initialized with results_dir: {results_dir}")
    
    def validate_predictions(self, prediction_results: List[Dict], actual_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate predictions against actual data
        
        Args:
            prediction_results: List of prediction results from backtester
            actual_data: DataFrame with actual market data
            
        Returns:
            Dictionary containing validation metrics and results
        """
        try:
            logger.info(f"Validating {len(prediction_results)} prediction results")
            
            # Prepare data for comparison
            validation_data = self._prepare_validation_data(prediction_results, actual_data)
            
            if validation_data is None or validation_data.empty:
                logger.warning("No validation data available")
                return {"error": "No validation data available"}
            
            # Calculate accuracy metrics
            metrics = self._calculate_accuracy_metrics(validation_data)
            
            # Generate validation plots
            plot_files = self._generate_validation_plots(validation_data, metrics)
            
            # Compile results
            results = {
                "validation_summary": {
                    "total_predictions": len(validation_data),
                    "validation_period": {
                        "start": validation_data['timestamp'].min(),
                        "end": validation_data['timestamp'].max()
                    },
                    "symbol": prediction_results[0].get('symbol', 'Unknown') if prediction_results else 'Unknown'
                },
                "accuracy_metrics": metrics,
                "plot_files": plot_files,
                "validation_data": validation_data.to_dict('records') if len(validation_data) < 1000 else "Large dataset - not included in JSON"
            }
            
            logger.info(f"Validation completed successfully. Accuracy: {metrics.get('overall_accuracy', {}).get('mape', 'N/A')}% MAPE")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating predictions: {e}")
            return {"error": str(e)}
    
    def _prepare_validation_data(self, prediction_results: List[Dict], actual_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare data for validation by aligning predictions with actual values
        """
        try:
            validation_records = []
            
            for pred_result in prediction_results:
                symbol = pred_result.get('symbol')
                pred_timestamp = pd.to_datetime(pred_result.get('prediction_timestamp'))
                predictions_str = pred_result.get('predictions', '')
                
                # Parse predictions DataFrame from string representation
                try:
                    if isinstance(predictions_str, pd.DataFrame):
                        pred_df = predictions_str
                    elif isinstance(predictions_str, str) and predictions_str.strip():
                        # Parse the string representation of DataFrame
                        pred_df = pd.read_csv(StringIO(predictions_str), index_col=0, parse_dates=True)
                    else:
                        pred_df = pd.DataFrame()
                except Exception as e:
                    logger.warning(f"Could not parse predictions for {pred_timestamp}: {e}")
                    continue
                
                if pred_df.empty:
                    continue
                
                logger.info(f"Processing prediction for {symbol} at {pred_timestamp}")
                logger.info(f"Prediction DataFrame shape: {pred_df.shape}")
                logger.info(f"Prediction timestamps: {pred_df.index.tolist()}")
                logger.info(f"Actual data shape: {actual_data.shape}")
                logger.info(f"Actual data range: {actual_data.index.min()} to {actual_data.index.max()}")
                
                # For backtesting, we need to find actual data that corresponds to the predicted timestamps
                # The actual data should contain the historical prices for the predicted time periods
                
                # Align predictions with actual data
                for _, pred_row in pred_df.iterrows():
                    pred_time = pd.to_datetime(pred_row.name)  # Index is timestamp
                    pred_price = pred_row['predicted_price']
                    
                    logger.debug(f"Looking for actual data for prediction at {pred_time}")
                    
                    # Find the closest actual data point for this predicted timestamp
                    if 'timestamp' in actual_data.columns:
                        time_diff = abs(actual_data['timestamp'] - pred_time)
                    else:
                        time_diff = abs(actual_data.index - pred_time)
                    
                    closest_idx = time_diff.argmin()
                    closest_time_diff = time_diff.min()
                    
                    logger.debug(f"Closest actual data at index {closest_idx}, time diff: {closest_time_diff}")
                    
                    # Use the closest data point if it's within a reasonable time window
                    if closest_time_diff <= timedelta(days=1):  # Within 1 day for daily predictions
                        # For predictions, we need to determine if this is predicting the opening or closing price
                        # If the prediction time is after market hours (after 16:00), it's predicting next day's opening
                        # If the prediction time is during market hours, it's predicting the closing price
                        
                        prediction_hour = pred_timestamp.hour
                        is_after_hours = prediction_hour >= 16  # After 4 PM
                        
                        if is_after_hours and pred_time.date() > pred_timestamp.date():
                            # This is predicting next day's opening price
                            actual_price = actual_data.iloc[closest_idx]['open']
                            price_type = "opening"
                        else:
                            # This is predicting the closing price for the same day
                            actual_price = actual_data.iloc[closest_idx]['close']
                            price_type = "closing"
                        
                        # Calculate time horizon correctly (predicted time - prediction time)
                        time_horizon = (pred_time - pred_timestamp).total_seconds() / 3600  # hours
                        
                        validation_records.append({
                            'timestamp': pred_time,
                            'predicted_price': pred_price,
                            'actual_price': actual_price,
                            'prediction_time': pred_timestamp,
                            'time_horizon': time_horizon,
                            'error': pred_price - actual_price,
                            'error_percent': ((pred_price - actual_price) / actual_price) * 100,
                            'price_type': price_type
                        })
                        
                        logger.info(f"Matched prediction {pred_time} with actual {actual_data.index[closest_idx]} "
                                   f"({price_type} price: {actual_price:.2f}, error: {pred_price - actual_price:.2f})")
                    else:
                        # Try to find data within 2 days for more lenient matching
                        if closest_time_diff <= timedelta(days=2):
                            # Apply same logic for opening vs closing price
                            prediction_hour = pred_timestamp.hour
                            is_after_hours = prediction_hour >= 16  # After 4 PM
                            
                            if is_after_hours and pred_time.date() > pred_timestamp.date():
                                actual_price = actual_data.iloc[closest_idx]['open']
                                price_type = "opening"
                            else:
                                actual_price = actual_data.iloc[closest_idx]['close']
                                price_type = "closing"
                            
                            time_horizon = (pred_time - pred_timestamp).total_seconds() / 3600
                            
                            validation_records.append({
                                'timestamp': pred_time,
                                'predicted_price': pred_price,
                                'actual_price': actual_price,
                                'prediction_time': pred_timestamp,
                                'time_horizon': time_horizon,
                                'error': pred_price - actual_price,
                                'error_percent': ((pred_price - actual_price) / actual_price) * 100,
                                'price_type': price_type
                            })
                            
                            logger.info(f"Matched prediction {pred_time} with actual {actual_data.index[closest_idx]} "
                                       f"({price_type} price: {actual_price:.2f}, error: {pred_price - actual_price:.2f}) - lenient match")
                        else:
                            logger.warning(f"No close match for prediction at {pred_time} "
                                         f"(closest: {actual_data.index[closest_idx]}, diff: {closest_time_diff})")
            
            if not validation_records:
                logger.warning("No validation records created")
                return None
            
            validation_df = pd.DataFrame(validation_records)
            validation_df = validation_df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Created {len(validation_df)} validation records")
            logger.info(f"Validation data sample:")
            logger.info(validation_df[['timestamp', 'predicted_price', 'actual_price', 'error']].head())
            
            return validation_df
            
        except Exception as e:
            logger.error(f"Error preparing validation data: {e}")
            return None
    
    def _calculate_accuracy_metrics(self, validation_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive accuracy metrics
        """
        try:
            predicted = validation_data['predicted_price'].values
            actual = validation_data['actual_price'].values
            
            # Basic metrics
            mae = np.mean(np.abs(predicted - actual))
            rmse = np.sqrt(np.mean((predicted - actual) ** 2))
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Directional accuracy (correctly predicted up/down movement)
            pred_direction = np.diff(predicted) > 0
            actual_direction = np.diff(actual) > 0
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
            
            # Correlation
            correlation = np.corrcoef(predicted, actual)[0, 1]
            
            # R-squared
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Time horizon analysis
            time_horizons = validation_data['time_horizon'].values
            horizon_metrics = {}
            
            for horizon in [1, 6, 24, 72, 168]:  # 1h, 6h, 1d, 3d, 1w
                horizon_mask = np.abs(time_horizons - horizon) <= 3  # Within 3 hours of target
                if np.any(horizon_mask):
                    horizon_pred = predicted[horizon_mask]
                    horizon_actual = actual[horizon_mask]
                    
                    horizon_mae = np.mean(np.abs(horizon_pred - horizon_actual))
                    horizon_mape = np.mean(np.abs((horizon_actual - horizon_pred) / horizon_actual)) * 100
                    
                    horizon_metrics[f"{horizon}h"] = {
                        "mae": float(horizon_mae),
                        "mape": float(horizon_mape),
                        "samples": int(np.sum(horizon_mask))
                    }
            
            metrics = {
                "overall_accuracy": {
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mape": float(mape),
                    "directional_accuracy": float(directional_accuracy),
                    "correlation": float(correlation),
                    "r_squared": float(r_squared)
                },
                "horizon_accuracy": horizon_metrics,
                "sample_size": len(validation_data)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            return {"error": str(e)}
    
    def _generate_validation_plots(self, validation_data: pd.DataFrame, metrics: Dict) -> List[str]:
        """
        Generate validation plots using matplotlib
        """
        try:
            plot_files = []
            symbol = "AAPL"  # Default, could be extracted from data
            
            # Set up the plotting style
            plt.style.use('default')
            fig_size = (15, 10)
            
            # 1. Predicted vs Actual Scatter Plot
            plt.figure(figsize=fig_size)
            plt.scatter(validation_data['actual_price'], validation_data['predicted_price'], 
                       alpha=0.6, s=20)
            
            # Perfect prediction line
            min_price = min(validation_data['actual_price'].min(), validation_data['predicted_price'].min())
            max_price = max(validation_data['actual_price'].max(), validation_data['predicted_price'].max())
            plt.plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2, label='Perfect Prediction')
            
            plt.xlabel('Actual Price ($)')
            plt.ylabel('Predicted Price ($)')
            plt.title(f'Predicted vs Actual Prices - {symbol}\nR² = {metrics["overall_accuracy"]["r_squared"]:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            scatter_file = os.path.join(self.plots_dir, f'{symbol}_predicted_vs_actual_scatter.png')
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(scatter_file)
            
            # 2. Time Series Comparison
            plt.figure(figsize=(20, 10))
            
            # Get historical data for context (previous 30 trading days)
            historical_data = self.actual_data_for_validation.copy()
            if not historical_data.empty:
                # Get the last 30 trading days before the first prediction
                first_prediction_time = validation_data['timestamp'].min()
                
                # Convert to datetime for proper filtering
                if isinstance(first_prediction_time, str):
                    first_prediction_time = pd.to_datetime(first_prediction_time)
                
                # Get data before the first prediction
                historical_before = historical_data[historical_data.index < first_prediction_time]
                
                # Resample to daily data and get the last 30 trading days
                # Use business days only to exclude weekends
                daily_data = historical_before.resample('B').last().dropna()  # 'B' = business days only
                
                # Get the last 30 business days before the prediction period
                if len(daily_data) >= 30:
                    historical_context = daily_data.tail(30)
                else:
                    # If we don't have enough business days, use all available days
                    historical_context = daily_data
                    logger.warning(f"Only {len(historical_context)} business days available for historical context")
                
                if len(historical_context) > 0:
                    # Plot historical context (previous 30 trading days)
                    plt.plot(historical_context.index, historical_context['close'], 
                            'g-', label='Historical Context (Previous 30 Days)', linewidth=2, alpha=0.7, marker='o', markersize=6)
                    
                    # Add vertical line to separate historical from prediction period
                    plt.axvline(x=first_prediction_time, color='black', linestyle='--', alpha=0.5, linewidth=2)
                    plt.text(first_prediction_time, plt.ylim()[1] * 0.95, 'Prediction Start', 
                            rotation=90, verticalalignment='top', fontsize=10, alpha=0.7)
                    
                    logger.info(f"Historical context: {len(historical_context)} days from {historical_context.index[0]} to {historical_context.index[-1]}")
            
            # Plot prediction period data
            sample_data = validation_data.copy()
            
            if len(sample_data) > 0:
                # Plot actual prices for prediction period
                plt.plot(sample_data['timestamp'], sample_data['actual_price'], 
                        'b-', label='Actual Price (Prediction Period)', linewidth=3, alpha=0.9, marker='o', markersize=8)
                
                # Plot predicted prices for prediction period
                plt.plot(sample_data['timestamp'], sample_data['predicted_price'], 
                        'r--', label='Predicted Price (Prediction Period)', linewidth=3, alpha=0.9, marker='s', markersize=8)
                
                # Add arrows showing the prediction flow (how each prediction feeds into the next)
                for i in range(len(sample_data) - 1):
                    # Draw arrow from prediction to next prediction to show data flow
                    plt.annotate('', xy=(sample_data['timestamp'].iloc[i+1], sample_data['predicted_price'].iloc[i+1]), 
                               xytext=(sample_data['timestamp'].iloc[i], sample_data['predicted_price'].iloc[i]),
                               arrowprops=dict(arrowstyle='->', color='red', alpha=0.5, lw=2))
                
                # Add text annotation explaining the data flow
                plt.text(0.02, 0.98, 'Red arrows show how each prediction feeds into the next prediction\n(simulating production data flow)', 
                        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            else:
                logger.warning("No data available for time series plot")
            
            plt.xlabel('Time')
            plt.ylabel('Price ($)')
            plt.title(f'Price Prediction Time Series - {symbol}\nMAPE: {metrics["overall_accuracy"]["mape"]:.2f}%')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Format x-axis dates and limit ticks
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            
            # Limit the number of ticks to avoid the warning
            ax.locator_params(axis='x', nbins=10)  # Limit to 10 ticks max
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            
            timeseries_file = os.path.join(self.plots_dir, f'{symbol}_timeseries_comparison.png')
            plt.savefig(timeseries_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(timeseries_file)
            
            # 3. Error Analysis
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Error distribution
            errors = validation_data['error'].values
            if len(errors) > 0:
                # Use fewer bins for small datasets
                bins = min(20, len(errors))
                ax1.hist(errors, bins=bins, alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Prediction Error ($)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Error Distribution')
                ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No error data available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Error Distribution')
            
            # Error over time
            ax2.scatter(validation_data['timestamp'], validation_data['error'], 
                       alpha=0.6, s=10)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Prediction Error ($)')
            ax2.set_title('Error Over Time')
            ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax2.grid(True, alpha=0.3)
            
            # Limit ticks for error over time plot
            ax2.locator_params(axis='x', nbins=10)
            
            # Error vs Time Horizon
            ax3.scatter(validation_data['time_horizon'], validation_data['error'], 
                       alpha=0.6, s=10)
            ax3.set_xlabel('Time Horizon (hours)')
            ax3.set_ylabel('Prediction Error ($)')
            ax3.set_title('Error vs Prediction Horizon')
            ax3.axhline(0, color='red', linestyle='--', alpha=0.7)
            ax3.grid(True, alpha=0.3)
            
            # Accuracy by Time Horizon
            if metrics.get('horizon_accuracy'):
                horizons = list(metrics['horizon_accuracy'].keys())
                horizon_mape = [metrics['horizon_accuracy'][h]['mape'] for h in horizons]
                horizon_labels = [h.replace('h', 'h') for h in horizons]
                
                ax4.bar(horizon_labels, horizon_mape, alpha=0.7)
                ax4.set_xlabel('Time Horizon')
                ax4.set_ylabel('MAPE (%)')
                ax4.set_title('Accuracy by Prediction Horizon')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            error_file = os.path.join(self.plots_dir, f'{symbol}_error_analysis.png')
            plt.savefig(error_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(error_file)
            
            # 4. Accuracy Summary Dashboard
            plt.figure(figsize=(12, 8))
            
            # Create a summary dashboard
            metrics_text = f"""
            ACCURACY SUMMARY - {symbol}
            
            Overall Performance:
            • MAPE: {metrics['overall_accuracy']['mape']:.2f}%
            • MAE: ${metrics['overall_accuracy']['mae']:.2f}
            • RMSE: ${metrics['overall_accuracy']['rmse']:.2f}
            • Directional Accuracy: {metrics['overall_accuracy']['directional_accuracy']:.1f}%
            • Correlation: {metrics['overall_accuracy']['correlation']:.3f}
            • R²: {metrics['overall_accuracy']['r_squared']:.3f}
            
            Sample Size: {metrics['sample_size']:,} predictions
            """
            
            plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title('Backtest Validation Summary', fontsize=16, fontweight='bold')
            
            summary_file = os.path.join(self.plots_dir, f'{symbol}_validation_summary.png')
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(summary_file)
            
            logger.info(f"Generated {len(plot_files)} validation plots")
            return plot_files
            
        except Exception as e:
            logger.error(f"Error generating validation plots: {e}")
            return []
    
    def save_validation_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Save validation results to file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"validation_results_{timestamp}.json"
            
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Validation results saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")
            return ""
