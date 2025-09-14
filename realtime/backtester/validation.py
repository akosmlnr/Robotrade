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
                
                # Check if we have enough future data for validation
                prediction_start = pred_df.index.min()
                prediction_end = pred_df.index.max()
                actual_data_after_prediction = actual_data[actual_data.index > prediction_end]
                logger.info(f"Prediction period: {prediction_start} to {prediction_end}")
                logger.info(f"Actual data after prediction: {len(actual_data_after_prediction)} records")
                if len(actual_data_after_prediction) > 0:
                    logger.info(f"Actual data after prediction range: {actual_data_after_prediction.index.min()} to {actual_data_after_prediction.index.max()}")
                
                # For backtesting, we need to find actual data that corresponds to the predicted timestamps
                # Since we're doing historical backtesting, we'll use the actual data that comes after the prediction period
                # This simulates how the model would perform on future data
                
                # Align predictions with actual data
                logger.info(f"Processing {len(pred_df)} predictions from prediction result at {pred_timestamp}")
                for idx, (_, pred_row) in enumerate(pred_df.iterrows()):
                    pred_time = pd.to_datetime(pred_row.name)  # Index is timestamp
                    pred_price = pred_row['predicted_price']
                    
                    if idx % 10 == 0:  # Log every 10th prediction
                        logger.info(f"Processing prediction {idx+1}/{len(pred_df)} at {pred_time}")
                    
                    # Find the closest actual data point for this predicted timestamp
                    if 'timestamp' in actual_data.columns:
                        time_diff = abs(actual_data['timestamp'] - pred_time)
                    else:
                        time_diff = abs(actual_data.index - pred_time)
                    
                    closest_idx = time_diff.argmin()
                    closest_time_diff = time_diff.min()
                    
                    logger.debug(f"Closest actual data at index {closest_idx}, time diff: {closest_time_diff}")
                    
                    # Use the closest data point if it's within a strict time window
                    # Only accept matches within 1 day for daily predictions to ensure accuracy
                    if closest_time_diff <= timedelta(days=1):  # Within 1 day for strict matching
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
                        # No lenient matching - only use strict matches
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
            
            # Debug: Check the time range of validation data
            if len(validation_df) > 0:
                time_range = validation_df['timestamp'].max() - validation_df['timestamp'].min()
                logger.info(f"Validation data time range: {time_range.days} days")
                logger.info(f"First prediction: {validation_df['timestamp'].min()}")
                logger.info(f"Last prediction: {validation_df['timestamp'].max()}")
                
                # Check if we have enough data for 10 weeks
                if time_range.days < 70:  # 10 weeks = 70 days
                    logger.warning(f"Validation data only covers {time_range.days} days, expected 70+ days for 10 weeks")
            
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
            historical_context = pd.DataFrame()  # Initialize as empty DataFrame
            
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
                
                logger.info(f"Historical data available: {len(historical_data)} records")
                logger.info(f"Historical before prediction: {len(historical_before)} records")
                logger.info(f"Daily data: {len(daily_data)} records")
                logger.info(f"Historical context: {len(historical_context)} records")
            
            if len(historical_context) > 0:
                # Convert historical data to numeric indices for consistent plotting
                historical_indices = list(range(-len(historical_context), 0))
                historical_prices = historical_context['close'].values
                
                # Plot historical context (previous 30 trading days)
                plt.plot(historical_indices, historical_prices, 
                        'g-', label='Historical Context (Previous 30 Days)', linewidth=2, alpha=0.7, marker='o', markersize=6)
                
                logger.info(f"Historical context plotted: {len(historical_context)} days from {historical_context.index[0]} to {historical_context.index[-1]}")
            else:
                logger.warning("No historical context data available for plotting")
            
            # Plot prediction period data
            sample_data = validation_data.copy()
            
            if len(sample_data) > 0:
                # Sort data by timestamp and convert to numeric indices
                sample_data = sample_data.sort_values('timestamp').reset_index(drop=True)
                prediction_indices = list(range(len(sample_data)))
                
                # Plot actual prices for prediction period
                plt.plot(prediction_indices, sample_data['actual_price'], 
                        'b-', label='Actual Price (Prediction Period)', linewidth=3, alpha=0.9, marker='o', markersize=8)
                
                # Plot predicted prices for prediction period
                plt.plot(prediction_indices, sample_data['predicted_price'], 
                        'r--', label='Predicted Price (Prediction Period)', linewidth=3, alpha=0.9, marker='s', markersize=8)
                
                # Add arrows showing the prediction flow (how each prediction feeds into the next)
                for i in range(len(sample_data) - 1):
                    # Draw arrow from prediction to next prediction to show data flow
                    plt.annotate('', xy=(i+1, sample_data['predicted_price'].iloc[i+1]), 
                               xytext=(i, sample_data['predicted_price'].iloc[i]),
                               arrowprops=dict(arrowstyle='->', color='red', alpha=0.5, lw=2))
                
                # Add vertical line to separate historical from prediction period
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
                plt.text(0, plt.ylim()[1] * 0.95, 'Prediction Start', 
                        rotation=90, verticalalignment='top', fontsize=10, alpha=0.7)
                
                logger.info(f"Prediction period: {len(sample_data)} days")
            else:
                logger.warning("No data available for time series plot")
            
            plt.xlabel('Time')
            plt.ylabel('Price ($)')
            plt.title(f'Price Prediction Time Series - {symbol}\nMAPE: {metrics["overall_accuracy"]["mape"]:.2f}%')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Format x-axis to show weeks
            ax = plt.gca()
            
            if len(sample_data) > 0:
                # Calculate number of weeks for the prediction period
                total_prediction_days = len(sample_data)
                num_weeks = max(1, (total_prediction_days + 6) // 7)  # Round up to get number of weeks
                
                # Ensure we show at least 10 weeks if we have enough data
                target_weeks = min(10, num_weeks)  # Show up to 10 weeks
                target_days = target_weeks * 7
                
                # Create tick positions and labels
                tick_positions = []
                tick_labels = []
                
                # Add tick for prediction start (index 0)
                tick_positions.append(0)
                tick_labels.append('Prediction Start')
                
                # Add weekly ticks for prediction period (up to 10 weeks)
                # Start from day 7 (end of first week) to get "1. week" label
                for i in range(7, min(target_days, total_prediction_days), 7):
                    tick_positions.append(i)
                    week_num = (i // 7)  # This gives us 1, 2, 3, etc. for weeks
                    tick_labels.append(f'{week_num}. week')
                
                # Add final tick only if we have more data than what's already covered
                if total_prediction_days > 1:
                    final_tick = min(target_days - 1, total_prediction_days - 1)
                    # Only add final tick if it's not already included and represents a new week
                    if final_tick not in tick_positions and final_tick > 0:
                        final_week = (final_tick // 7)  # This gives us 1, 2, 3, etc. for weeks
                        # Only add if it's a different week than the last one
                        if not tick_labels or f'{final_week}. week' != tick_labels[-1]:
                            tick_positions.append(final_tick)
                            tick_labels.append(f'{final_week}. week')
                
                # Set the ticks and labels
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)
                
                # Set x-axis limits to show both historical and prediction data
                historical_days = len(historical_context) if len(historical_context) > 0 else 0
                ax.set_xlim(-historical_days - 1, min(target_days, total_prediction_days) + 1)
                
                logger.info(f"X-axis: {len(tick_positions)} ticks, {target_weeks} weeks target, {total_prediction_days} prediction days available")
                logger.info(f"Historical days: {historical_days}, X-axis range: {-historical_days - 1} to {min(target_days, total_prediction_days) + 1}")
            else:
                # Fallback formatting
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax.locator_params(axis='x', nbins=10)
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
