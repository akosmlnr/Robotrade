"""
Prediction Simulator for Backtesting
Simulates real-time predictions with incremental data updates
"""

import time
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
import queue
import os
import sys

# Add parent directory to path to import realtime modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prediction_engine import PredictionEngine
from models.model_manager import ModelManager
from storage.data_storage import DataStorage

logger = logging.getLogger(__name__)

class PredictionSimulator:
    """
    Simulates real-time predictions by running predictions every second
    with incremental 15-minute data updates for backtesting
    """
    
    def __init__(self, model_manager: ModelManager, data_storage: DataStorage,
                 simulation_speed: float = 1.0):
        """
        Initialize the prediction simulator
        
        Args:
            model_manager: ModelManager instance
            data_storage: DataStorage instance  
            simulation_speed: Speed multiplier for simulation (1.0 = real-time, 2.0 = 2x speed)
        """
        self.model_manager = model_manager
        self.data_storage = data_storage
        self.simulation_speed = simulation_speed
        self.prediction_engine = PredictionEngine(model_manager, data_storage)
        
        # Simulation state
        self.is_running = False
        self.current_symbol = None
        self.current_data_updates = []
        self.current_update_index = 0
        self.simulation_start_time = None
        self.actual_start_time = None
        
        # Results tracking
        self.prediction_history = []
        self.accuracy_metrics = {}
        
        # Validation data storage
        self.actual_data_for_validation = []
        
        # Threading
        self.simulation_thread = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.on_prediction_callback = None
        self.on_update_callback = None
        self.on_completion_callback = None
        
        logger.info(f"PredictionSimulator initialized with speed: {simulation_speed}x")
    
    def set_callbacks(self, on_prediction: Callable = None, 
                     on_update: Callable = None, 
                     on_completion: Callable = None):
        """
        Set callback functions for simulation events
        
        Args:
            on_prediction: Called after each prediction is made
            on_update: Called after each data update
            on_completion: Called when simulation completes
        """
        self.on_prediction_callback = on_prediction
        self.on_update_callback = on_update
        self.on_completion_callback = on_completion
        
        logger.info("Callbacks set for prediction simulator")
    
    def start_simulation(self, symbol: str, data_updates: List[Dict[str, Any]]) -> bool:
        """
        Start the prediction simulation
        
        Args:
            symbol: Stock symbol to simulate
            data_updates: List of incremental data updates
            
        Returns:
            True if simulation started successfully, False otherwise
        """
        try:
            if self.is_running:
                logger.warning("Simulation already running, stopping current simulation")
                self.stop_simulation()
            
            if not data_updates:
                logger.error("No data updates provided for simulation")
                return False
            
            # Validate model availability
            model_data = self.model_manager.get_model(symbol)
            if not model_data:
                logger.error(f"No model available for symbol {symbol}")
                return False
            
            # Initialize simulation state
            self.current_symbol = symbol
            self.current_data_updates = data_updates
            self.current_update_index = 0
            self.simulation_start_time = datetime.now()
            self.actual_start_time = data_updates[0]['timestamp']
            self.prediction_history = []
            self.accuracy_metrics = {}
            self.stop_event.clear()
            
            logger.info(f"Starting simulation for {symbol} with {len(data_updates)} data updates")
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(
                target=self._run_simulation_loop, 
                daemon=True
            )
            self.simulation_thread.start()
            
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Error starting simulation for {symbol}: {e}")
            return False
    
    def stop_simulation(self):
        """Stop the current simulation"""
        try:
            if not self.is_running:
                return
            
            logger.info("Stopping prediction simulation")
            self.stop_event.set()
            self.is_running = False
            
            # Wait for simulation thread to finish
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.simulation_thread.join(timeout=5.0)
            
            # Call completion callback
            if self.on_completion_callback:
                try:
                    self.on_completion_callback(self._get_simulation_results())
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")
            
            logger.info("Prediction simulation stopped")
            
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
    
    def _run_simulation_loop(self):
        """Main simulation loop that runs in a separate thread"""
        try:
            logger.info("Starting simulation loop")
            
            # Process all updates instantly - no delays
            total_updates = len(self.current_data_updates)
            logger.info(f"Processing {total_updates} updates for weekly prediction")
            
            # Make only ONE prediction at the start of the week
            # Use the first update that has enough data for a meaningful prediction
            prediction_made = False
            
            for i, current_update in enumerate(self.current_data_updates):
                if self.stop_event.is_set():
                    break
                    
                self.current_update_index = i
                
                # Only make prediction once, when we have enough data
                if not prediction_made and len(current_update['available_data']) >= 25:  # Need at least 25 data points
                    prediction_result = self._make_weekly_prediction(current_update)
                    prediction_made = True
                    
                    if prediction_result:
                        self.prediction_history.append(prediction_result)
                        
                        # Call prediction callback
                        if self.on_prediction_callback:
                            try:
                                self.on_prediction_callback(prediction_result)
                            except Exception as e:
                                logger.error(f"Error in prediction callback: {e}")
                
                # Call update callback
                if self.on_update_callback:
                    try:
                        self.on_update_callback(current_update, prediction_result if prediction_made else None)
                    except Exception as e:
                        logger.error(f"Error in update callback: {e}")
                
                # Log progress every 100 updates
                if (i + 1) % 100 == 0 or i == total_updates - 1:
                    logger.info(f"Processed {i + 1}/{total_updates} updates ({len(self.prediction_history)} total predictions)")
            
            logger.info("Completed processing all updates for weekly prediction")
            
            # Calculate final accuracy metrics
            self._calculate_accuracy_metrics()
            
            # Mark simulation as completed
            self.is_running = False
            logger.info("Simulation loop completed")
            
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            self.is_running = False
        finally:
            # Ensure simulation is marked as stopped
            self.is_running = False
            logger.info("Simulation loop cleanup completed")
    
    def _make_weekly_prediction(self, data_update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make sequential daily predictions that simulate production behavior
        
        Args:
            data_update: Current data update containing available data
            
        Returns:
            Prediction result dictionary or None if failed
        """
        try:
            symbol = self.current_symbol
            available_data = data_update['available_data']
            current_timestamp = data_update['timestamp']
            
            # Validate data update
            if available_data.empty:
                logger.warning(f"No data available for {symbol} at {current_timestamp}")
                return None
            
            # Check if we have enough data for prediction
            model_data = self.model_manager.get_model(symbol)
            if not model_data:
                logger.warning(f"No model available for {symbol}")
                return None
            
            sequence_length = model_data['config']['sequence_length']
            
            if len(available_data) < sequence_length:
                logger.debug(f"Insufficient data for {symbol}: need {sequence_length}, got {len(available_data)}")
                return None
            
            # Get the full historical data for sequential predictions
            if not hasattr(self, 'full_historical_data') or self.full_historical_data.empty:
                logger.warning("No full historical data available for sequential predictions")
                return None
            
            # Generate sequential daily predictions (simulating production workflow)
            # Only predict for TRADING DAYS (Monday-Friday), not weekends
            daily_predictions = []
            daily_timestamps = []
            prediction_timestamps = []
            
            # Start from the next trading day after the prediction timestamp
            prediction_start = current_timestamp.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
            
            # Find the next trading day (skip weekends)
            while prediction_start.weekday() >= 5:  # 5=Saturday, 6=Sunday
                prediction_start += timedelta(days=1)
            
            logger.info(f"Making sequential predictions for {symbol} starting from {prediction_start} (trading days only)")
            
            # Create a rolling dataset that includes previous predictions' results
            # This simulates how the model would work in production with updated data
            rolling_data = self.full_historical_data.copy()
            
            # Predict for up to 30 trading days (six weeks of trading)
            trading_days_predicted = 0
            current_prediction_day = prediction_start
            
            while trading_days_predicted < 30:  # Maximum 30 trading days (six weeks)
                # Skip weekends
                if current_prediction_day.weekday() >= 5:  # Saturday or Sunday
                    current_prediction_day += timedelta(days=1)
                    continue
                
                try:
                    # Calculate the prediction time (end of previous trading day)
                    # For ALL predictions, use the end of the previous trading day for consistency
                    if trading_days_predicted == 0:
                        # For the first prediction, use the end of the previous trading day
                        # This ensures continuity with historical data
                        prediction_time = current_prediction_day - timedelta(days=1)
                        prediction_time = prediction_time.replace(hour=16, minute=0, second=0)  # End of trading day
                        
                        # Skip weekends for prediction time
                        while prediction_time.weekday() >= 5:
                            prediction_time -= timedelta(days=1)
                            prediction_time = prediction_time.replace(hour=16, minute=0, second=0)
                    else:
                        # For subsequent predictions, use the end of the previous trading day
                        prediction_time = current_prediction_day - timedelta(days=1)
                        prediction_time = prediction_time.replace(hour=16, minute=0, second=0)  # End of trading day
                        
                        # Skip weekends for prediction time too
                        while prediction_time.weekday() >= 5:
                            prediction_time -= timedelta(days=1)
                            prediction_time = prediction_time.replace(hour=16, minute=0, second=0)
                    
                    # Get data up to the prediction time (simulating end-of-day prediction)
                    historical_data_up_to_prediction = rolling_data[
                        rolling_data.index <= prediction_time
                    ].copy()
                    
                    if len(historical_data_up_to_prediction) < sequence_length:
                        logger.warning(f"Insufficient data for trading day {trading_days_predicted + 1} prediction: need {sequence_length}, got {len(historical_data_up_to_prediction)}")
                        # Use the last available price as fallback
                        if not historical_data_up_to_prediction.empty:
                            fallback_price = historical_data_up_to_prediction['close'].iloc[-1]
                        else:
                            fallback_price = available_data['close'].iloc[-1]
                        daily_predictions.append(fallback_price)
                        daily_timestamps.append(current_prediction_day)
                        prediction_timestamps.append(prediction_time)
                        trading_days_predicted += 1
                        current_prediction_day += timedelta(days=1)
                        continue
                    
                    # Create prediction sequence using data up to prediction time
                    prediction_sequence = self.prediction_engine.create_prediction_sequence(
                        historical_data_up_to_prediction, 
                        sequence_length, 
                        model_data['config']['features'], 
                        model_data['scaler']
                    )
                    
                    if prediction_sequence is None:
                        logger.warning(f"Failed to create prediction sequence for day {trading_days_predicted + 1}")
                        # Use the last available price as fallback
                        fallback_price = historical_data_up_to_prediction['close'].iloc[-1]
                        daily_predictions.append(fallback_price)
                        daily_timestamps.append(current_prediction_day)
                        prediction_timestamps.append(prediction_time)
                        trading_days_predicted += 1
                        current_prediction_day += timedelta(days=1)
                        continue
                    
                    # Make prediction for the next day's opening price
                    model = model_data['model']
                    scaler = model_data['scaler']
                    prediction_scaled = model.predict(prediction_sequence, verbose=0)
                    prediction_price = scaler.inverse_transform(prediction_scaled)[0][0]
                    
                    # Validate prediction
                    current_price = historical_data_up_to_prediction['close'].iloc[-1]
                    if np.isnan(prediction_price) or np.isinf(prediction_price) or prediction_price <= 0:
                        prediction_price = current_price
                    
                    # For the first prediction, ensure continuity by anchoring to the last known price
                    # This simulates production behavior where predictions are incremental adjustments
                    if trading_days_predicted == 0:
                        # Calculate the price change from the model prediction
                        price_change = prediction_price - current_price
                        
                        # For the first prediction, apply a smaller, more realistic change
                        # This ensures continuity while still using the model's directional signal
                        realistic_change = price_change * 0.3  # Use 30% of the predicted change
                        prediction_price = current_price + realistic_change
                        
                        logger.debug(f"First prediction: current=${current_price:.2f}, raw_pred=${scaler.inverse_transform(prediction_scaled)[0][0]:.2f}, "
                                   f"change=${price_change:.2f}, realistic_change=${realistic_change:.2f}, final=${prediction_price:.2f}")
                    
                    # NO ARTIFICIAL NOISE - let the model's natural predictions drive results
                    daily_predictions.append(prediction_price)
                    daily_timestamps.append(current_prediction_day)
                    prediction_timestamps.append(prediction_time)
                    
                    # CRITICAL: Update the rolling dataset with the prediction result
                    # This simulates how the model would work in production with updated data
                    if trading_days_predicted > 0:  # Don't update for the first prediction
                        # Add the prediction as a new data point for the next prediction
                        prediction_timestamp = current_prediction_day.replace(hour=9, minute=30, second=0)
                        
                        # Create a new row with the predicted price
                        new_row = pd.DataFrame({
                            'close': [prediction_price],
                            'open': [prediction_price],  # Use prediction as open price
                            'high': [prediction_price * 1.01],  # Add small variation
                            'low': [prediction_price * 0.99],   # Add small variation
                            'volume': [historical_data_up_to_prediction['volume'].iloc[-1]]  # Use last volume
                        }, index=[prediction_timestamp])
                        
                        # Add to rolling data for next prediction
                        rolling_data = pd.concat([rolling_data, new_row])
                        rolling_data = rolling_data.sort_index()
                        
                        logger.debug(f"Updated rolling data with prediction: {prediction_timestamp} = ${prediction_price:.2f}")
                    
                    logger.debug(f"Trading day {trading_days_predicted + 1} prediction: ${prediction_price:.2f} (using data up to {prediction_time})")
                    
                    trading_days_predicted += 1
                    current_prediction_day += timedelta(days=1)
                    
                except Exception as e:
                    logger.warning(f"Error predicting trading day {trading_days_predicted + 1}: {e}")
                    # Use the last available price as fallback
                    if not historical_data_up_to_prediction.empty:
                        fallback_price = historical_data_up_to_prediction['close'].iloc[-1]
                    else:
                        fallback_price = available_data['close'].iloc[-1]
                    daily_predictions.append(fallback_price)
                    daily_timestamps.append(current_prediction_day)
                    prediction_timestamps.append(prediction_time)
                    
                    trading_days_predicted += 1
                    current_prediction_day += timedelta(days=1)
            
            # Create prediction DataFrame
            prediction_df = pd.DataFrame({
                'timestamp': daily_timestamps,
                'predicted_price': daily_predictions
            })
            prediction_df.set_index('timestamp', inplace=True)
            
            # Calculate confidence score based on prediction consistency
            confidence_score = self.prediction_engine.calculate_prediction_confidence(
                symbol, daily_predictions, available_data
            )
            
            # Create result
            result = {
                'symbol': symbol,
                'prediction_timestamp': current_timestamp,
                'predictions': prediction_df,
                'confidence_score': confidence_score,
                'sequence_length': sequence_length,
                'total_predictions': len(daily_predictions),
                'prediction_horizon': f'2 weeks ({len(daily_predictions)} trading days)',
                'data_points': len(daily_predictions),
                'available_data_points': len(available_data),
                'latest_actual_price': available_data['close'].iloc[-1],
                'predicted_week_end_price': daily_predictions[-1] if daily_predictions else available_data['close'].iloc[-1],
                'simulation_time': datetime.now(),
                'simulation_step': self.current_update_index,
                'prediction_method': 'sequential_trading_days_simulation'
            }
            
            logger.info(f"Generated sequential 2-week prediction for {symbol} at {current_timestamp}: "
                       f"${available_data['close'].iloc[-1]:.2f} -> ${daily_predictions[-1]:.2f} "
                       f"(change: {((daily_predictions[-1] - available_data['close'].iloc[-1]) / available_data['close'].iloc[-1] * 100):+.2f}%)")
            
            # Store actual data for validation - store the full historical data
            if not self.full_historical_data.empty:
                for timestamp, row in self.full_historical_data.iterrows():
                    self.actual_data_for_validation.append({
                        'timestamp': timestamp,
                        'close': row['close'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'volume': row['volume']
                    })
                
                logger.info(f"Stored {len(self.full_historical_data)} actual data points for validation")
                logger.info(f"Data range: {self.full_historical_data.index.min()} to {self.full_historical_data.index.max()}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making sequential weekly prediction for {symbol}: {e}")
            return None

    def _make_prediction(self, data_update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a prediction based on the current data update
        
        Args:
            data_update: Current data update containing available data
            
        Returns:
            Prediction result dictionary or None if failed
        """
        try:
            symbol = self.current_symbol
            available_data = data_update['available_data']
            current_timestamp = data_update['timestamp']
            
            # Validate data update
            if available_data.empty:
                logger.warning(f"No data available for {symbol} at {current_timestamp}")
                return None
            
            # Check if we have enough data for prediction
            model_data = self.model_manager.get_model(symbol)
            if not model_data:
                logger.warning(f"No model available for {symbol}")
                return None
            
            sequence_length = model_data['config']['sequence_length']
            
            if len(available_data) < sequence_length:
                logger.debug(f"Insufficient data for {symbol}: need {sequence_length}, got {len(available_data)}")
                return None
            
            # Create prediction sequence
            prediction_sequence = self.prediction_engine.create_prediction_sequence(
                available_data, sequence_length, model_data['config']['features'], model_data['scaler']
            )
            
            if prediction_sequence is None:
                logger.warning(f"Failed to create prediction sequence for {symbol}")
                return None
            
            # Generate full week of predictions (672 15-minute intervals) - INSTANT
            weekly_intervals = 7 * 24 * 4  # 672 predictions
            predictions = []
            timestamps = []
            
            # Start from the next 15-minute interval after current time
            current_time = current_timestamp + timedelta(minutes=15)
            current_price = available_data['close'].iloc[-1]
            
            # Make ONE prediction and extrapolate for speed (much faster than 672 individual predictions)
            try:
                # Make single prediction
                model = model_data['model']
                scaler = model_data['scaler']
                prediction_scaled = model.predict(prediction_sequence, verbose=0)
                prediction_price = scaler.inverse_transform(prediction_scaled)[0][0]
                
                # Validate prediction quickly
                if np.isnan(prediction_price) or np.isinf(prediction_price) or prediction_price <= 0:
                    prediction_price = current_price
                
                # Generate trend for the week based on single prediction
                price_change = prediction_price - current_price
                
                # Create 672 predictions with slight trend variation (instant)
                for i in range(weekly_intervals):
                    # Add small random variation to make it realistic
                    variation = np.random.normal(0, abs(price_change) * 0.1)  # 10% variation
                    predicted_price = prediction_price + variation
                    
                    # Ensure price stays reasonable
                    if predicted_price <= 0:
                        predicted_price = current_price
                    
                    predictions.append(predicted_price)
                    timestamps.append(current_time)
                    current_time += timedelta(minutes=15)
                    
            except Exception as e:
                logger.debug(f"Error in instant prediction for {symbol}: {e}")
                # Fallback: fill with current price
                for i in range(weekly_intervals):
                    predictions.append(current_price)
                    timestamps.append(current_timestamp + timedelta(minutes=15 * (i + 1)))
            
            # Create prediction DataFrame
            prediction_df = pd.DataFrame({
                'timestamp': timestamps,
                'predicted_price': predictions
            })
            prediction_df.set_index('timestamp', inplace=True)
            
            # Calculate confidence score
            confidence_score = self.prediction_engine.calculate_prediction_confidence(
                symbol, predictions, available_data
            )
            
            # Create result
            result = {
                'symbol': symbol,
                'prediction_timestamp': current_timestamp,
                'predictions': prediction_df,
                'confidence_score': confidence_score,
                'sequence_length': sequence_length,
                'total_predictions': len(predictions),
                'prediction_horizon': '1 week',
                'data_points': weekly_intervals,
                'available_data_points': len(available_data),
                'latest_actual_price': available_data['close'].iloc[-1],
                'simulation_time': datetime.now(),
                'simulation_step': self.current_update_index
            }
            
            logger.debug(f"Generated prediction for {symbol} at {current_timestamp} "
                        f"(step {self.current_update_index}/{len(self.current_data_updates)})")
            
            # Store actual data for validation
            self.actual_data_for_validation.extend([
                {
                    'timestamp': timestamp,
                    'close': row['close'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'volume': row['volume']
                }
                for timestamp, row in available_data.iterrows()
            ])
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return None
    
    def _calculate_accuracy_metrics(self):
        """Calculate accuracy metrics for the simulation"""
        try:
            if not self.prediction_history:
                return
            
            symbol = self.current_symbol
            actual_data = self.current_data_updates[-1]['available_data'] if self.current_data_updates else pd.DataFrame()
            
            if actual_data.empty:
                logger.warning("No actual data available for accuracy calculation")
                return
            
            # Calculate metrics for each prediction
            accuracy_results = []
            
            for pred_result in self.prediction_history:
                predictions_df = pred_result['predictions']
                pred_timestamp = pred_result['prediction_timestamp']
                
                # Find actual prices for the predicted time periods
                actual_prices = []
                predicted_prices = []
                
                for pred_time, pred_row in predictions_df.iterrows():
                    # Find closest actual price
                    time_diff = np.abs(actual_data.index - pred_time)
                    closest_idx = time_diff.argmin()
                    closest_time_diff = time_diff.min()
                    
                    # Only use predictions within reasonable time window (1 hour)
                    if closest_time_diff <= timedelta(hours=1):
                        actual_prices.append(actual_data.iloc[closest_idx]['close'])
                        predicted_prices.append(pred_row['predicted_price'])
                
                if actual_prices and predicted_prices:
                    # Handle division by zero and NaN values
                    actual_array = np.array(actual_prices)
                    predicted_array = np.array(predicted_prices)
                    
                    # Filter out any NaN or infinite values
                    valid_mask = np.isfinite(actual_array) & np.isfinite(predicted_array) & (actual_array > 0)
                    if not np.any(valid_mask):
                        logger.warning("No valid data points for accuracy calculation")
                        continue
                    
                    actual_array = actual_array[valid_mask]
                    predicted_array = predicted_array[valid_mask]
                    
                    mape = np.mean(np.abs((actual_array - predicted_array) / actual_array)) * 100
                    mae = np.mean(np.abs(actual_array - predicted_array))
                    rmse = np.sqrt(np.mean((actual_array - predicted_array) ** 2))
                    
                    accuracy_results.append({
                        'prediction_timestamp': pred_timestamp,
                        'mape': mape,
                        'mae': mae,
                        'rmse': rmse,
                        'confidence_score': pred_result['confidence_score'],
                        'data_points_used': len(actual_prices)
                    })
            
            # Calculate overall metrics
            if accuracy_results:
                self.accuracy_metrics = {
                    'overall_mape': np.mean([r['mape'] for r in accuracy_results]),
                    'overall_mae': np.mean([r['mae'] for r in accuracy_results]),
                    'overall_rmse': np.mean([r['rmse'] for r in accuracy_results]),
                    'average_confidence': np.mean([r['confidence_score'] for r in accuracy_results]),
                    'total_predictions': len(accuracy_results),
                    'detailed_results': accuracy_results
                }
                
                logger.info(f"Calculated accuracy metrics for {symbol}: "
                           f"MAPE={self.accuracy_metrics['overall_mape']:.2f}%, "
                           f"MAE=${self.accuracy_metrics['overall_mae']:.2f}, "
                           f"RMSE=${self.accuracy_metrics['overall_rmse']:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        try:
            status = {
                'is_running': self.is_running,
                'current_symbol': self.current_symbol,
                'total_updates': len(self.current_data_updates) if self.current_data_updates else 0,
                'current_update_index': self.current_update_index,
                'progress_percent': (self.current_update_index / len(self.current_data_updates) * 100) if self.current_data_updates else 0,
                'simulation_start_time': self.simulation_start_time,
                'elapsed_time': (datetime.now() - self.simulation_start_time).total_seconds() if self.simulation_start_time else 0,
                'total_predictions': len(self.prediction_history),
                'accuracy_metrics': self.accuracy_metrics
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting simulation status: {e}")
            return {'error': str(e)}
    
    def _get_simulation_results(self) -> Dict[str, Any]:
        """Get complete simulation results"""
        try:
            results = {
                'symbol': self.current_symbol,
                'simulation_start_time': self.simulation_start_time,
                'simulation_end_time': datetime.now(),
                'total_duration_seconds': (datetime.now() - self.simulation_start_time).total_seconds() if self.simulation_start_time else 0,
                'total_data_updates': len(self.current_data_updates) if self.current_data_updates else 0,
                'total_predictions': len(self.prediction_history),
                'accuracy_metrics': self.accuracy_metrics,
                'prediction_history': self.prediction_history,
                'final_status': self.get_simulation_status()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting simulation results: {e}")
            return {'error': str(e)}
    
    def get_prediction_history(self) -> List[Dict[str, Any]]:
        """Get the prediction history"""
        return self.prediction_history.copy()
    
    def get_accuracy_metrics(self) -> Dict[str, Any]:
        """Get the accuracy metrics"""
        return self.accuracy_metrics.copy()
    
    def get_actual_data_for_validation(self) -> pd.DataFrame:
        """
        Get actual market data collected during simulation for validation
        
        Returns:
            DataFrame with actual market data
        """
        if not self.actual_data_for_validation:
            return pd.DataFrame()
        
        # Convert to DataFrame and remove duplicates
        df = pd.DataFrame(self.actual_data_for_validation)
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # Set timestamp as index for consistency with validation system
        df.set_index('timestamp', inplace=True)
        
        return df


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # This would be used with actual ModelManager and DataStorage instances
    print("PredictionSimulator module loaded successfully")
    print("Use with ModelManager and DataStorage instances for full functionality")
