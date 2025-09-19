"""
Prediction Engine for Real-time LSTM Prediction System
Phase 1.3.1: 2-Week Ahead Prediction Logic with Rolling Window Updates
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class PredictionEngine:
    """
    Core prediction engine for generating 2-week ahead predictions using pre-trained LSTM models
    """
    
    def __init__(self, model_manager, data_storage):
        """
        Initialize the prediction engine
        
        Args:
            model_manager: ModelManager instance
            data_storage: DataStorage instance
        """
        self.model_manager = model_manager
        self.data_storage = data_storage
        self.prediction_cache = {}  # Cache for recent predictions
        self.feature_engineer = FeatureEngineer()  # Feature engineering
        
        logger.info("PredictionEngine initialized")
    
    def generate_weekly_prediction(self, symbol: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate 2-week ahead prediction for a symbol
        
        Args:
            symbol: Stock symbol
            model_data: Dictionary containing model, scaler, and config
            
        Returns:
            Dictionary with prediction results
        """
        try:
            model = model_data['model']
            scaler = model_data['scaler']
            config = model_data['config']
            
            # Prepare data for prediction
            sequence_length = config['sequence_length']
            
            # Get comprehensive historical data for better predictions
            # Use all available data instead of just recent data
            latest_data = self.data_storage.get_all_available_data(symbol)
            
            if latest_data.empty:
                logger.error(f"No market data available for {symbol}")
                return None
            
            logger.info(f"Retrieved {len(latest_data)} records for {symbol} prediction (need {sequence_length})")
            
            # Add features using JordiCorbilla approach
            data_with_features = self.feature_engineer.add_features(latest_data)
            features = self.feature_engineer.get_feature_names()
            
            logger.info(f"Generated {len(features)} features: {features[:10]}...")  # Show first 10 features
            
            # Create prediction sequence using proper features
            prediction_sequence = self.create_prediction_sequence(
                data_with_features, sequence_length, features, scaler
            )
            
            if prediction_sequence is None:
                logger.error(f"Failed to create prediction sequence for {symbol}")
                return None
            
            # Generate 2-week prediction (14 trading days)
            prediction_days = 14
            predictions = []
            timestamps = []
            
            # Start from next trading day
            current_time = latest_data.index[-1].replace(hour=16, minute=0, second=0, microsecond=0) + timedelta(days=1)
            current_time = self._get_next_trading_day(current_time)
            
            # Use the last known price as starting point
            current_price = latest_data['close'].iloc[-1]
            logger.info(f"Starting prediction from price: ${current_price:.2f} at {current_time}")
            
            # Safety check for valid current price
            if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
                logger.error(f"Invalid current price for {symbol}: {current_price}")
                return None
            
            # Generate predictions step by step (daily predictions)
            # Use consistent historical data for all predictions to maintain stability
            base_historical_data = data_with_features.copy()
            
            for i in range(prediction_days):
                try:
                    # Use consistent historical data for all predictions to maintain stability
                    # This matches the backtester approach for more stable predictions
                    prediction_data = base_historical_data.tail(sequence_length * 2)
                    
                    # Ensure we have enough data
                    if len(prediction_data) < sequence_length:
                        prediction_data = base_historical_data.tail(sequence_length * 2)
                    
                    # Debug logging for consistent data usage
                    if i < 5:  # Log first few predictions for debugging
                        data_start_price = prediction_data['close'].iloc[0] if len(prediction_data) > 0 else 0
                        data_end_price = prediction_data['close'].iloc[-1] if len(prediction_data) > 0 else 0
                        logger.debug(f"Day {i+1}: Using {len(prediction_data)} consistent data points, price range: ${data_start_price:.2f} to ${data_end_price:.2f}")
                    
                    # Create prediction sequence from consistent historical data
                    prediction_sequence = self.create_prediction_sequence(
                        prediction_data, sequence_length, features, scaler
                    )
                    
                    if prediction_sequence is None:
                        logger.warning(f"Failed to create prediction sequence for day {i+1}")
                        # Use the last available price as fallback
                        fallback_price = base_historical_data['close'].iloc[-1]
                        predictions.append(fallback_price)
                        timestamps.append(current_time)
                        current_time += timedelta(days=1)
                        continue
                    
                    # Make prediction for next day
                    prediction_scaled = model.predict(prediction_sequence, verbose=0)
                    prediction_price = scaler.inverse_transform(prediction_scaled)[0][0]
                    
                    # Debug logging for first few predictions
                    if i < 5:
                        logger.info(f"Day {i+1}: Scaled prediction: {prediction_scaled[0][0]:.6f}, "
                                   f"Inverse transform: {prediction_price:.2f}, "
                                   f"Historical price: {base_historical_data['close'].iloc[-1]:.2f}")
                    
                    # Validate prediction
                    if not self.validate_prediction_output(prediction_price):
                        logger.warning(f"Invalid prediction for {symbol} at day {i+1}")
                        # Use historical price as fallback
                        prediction_price = base_historical_data['close'].iloc[-1]
                    
                    # Apply realistic change factor for first prediction (like backtester)
                    if i == 0:
                        current_price = base_historical_data['close'].iloc[-1]
                        price_change = prediction_price - current_price
                        # Use smaller factor for more stable predictions like backtester
                        realistic_change = price_change * 0.3  # Use 30% of the predicted change for stable predictions
                        prediction_price = current_price + realistic_change
                        
                        logger.debug(f"First prediction: current=${current_price:.2f}, raw_pred=${scaler.inverse_transform(prediction_scaled)[0][0]:.2f}, "
                                   f"change=${price_change:.2f}, realistic_change=${realistic_change:.2f}, final=${prediction_price:.2f}")
                    
                    # No artificial volatility - let the model's natural predictions drive results
                    # This matches the backtester approach for more stable predictions
                    
                    predictions.append(prediction_price)
                    timestamps.append(current_time)
                    
                    # Move to next trading day (no rolling updates - using consistent historical data)
                    current_time = self._get_next_trading_day(current_time + timedelta(days=1))
                    
                except Exception as e:
                    logger.error(f"Error in prediction step {i} for {symbol}: {e}")
                    # Use historical price as fallback
                    fallback_price = base_historical_data['close'].iloc[-1]
                    predictions.append(fallback_price)
                    timestamps.append(current_time)
                    current_time += timedelta(days=1)
            
            # Create prediction DataFrame
            prediction_df = pd.DataFrame({
                'timestamp': timestamps,
                'predicted_price': predictions
            })
            prediction_df.set_index('timestamp', inplace=True)
            
            # Calculate confidence score
            confidence_score = self.calculate_prediction_confidence(
                symbol, predictions, latest_data
            )
            
            # Store predictions in database with time-based confidence decay
            self.store_predictions_with_decay(symbol, prediction_df, confidence_score)
            
            result = {
                'symbol': symbol,
                'prediction_timestamp': datetime.now(),
                'predictions': prediction_df,
                'confidence_score': confidence_score,
                'sequence_length': sequence_length,
                'total_predictions': len(predictions),
                'prediction_horizon': '2 weeks',
                'data_points': prediction_days
            }
            
            # Log prediction statistics
            pred_prices = np.array(predictions)
            price_std = np.std(pred_prices)
            price_range = np.max(pred_prices) - np.min(pred_prices)
            
            logger.info(f"Generated {len(predictions)} predictions for {symbol} with confidence {confidence_score:.3f}")
            logger.info(f"Prediction statistics: range=${price_range:.6f}, std=${price_std:.6f}, "
                       f"min=${np.min(pred_prices):.2f}, max=${np.max(pred_prices):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating 2-week prediction for {symbol}: {e}")
            return None
    
    def create_prediction_sequence(self, data: pd.DataFrame, sequence_length: int, 
                                 features: List[str], scaler) -> Optional[np.ndarray]:
        """
        Create input sequence for prediction
        
        Args:
            data: Market data DataFrame
            sequence_length: Number of time steps for sequence
            features: List of feature names
            scaler: Fitted scaler
            
        Returns:
            Prepared sequence array or None if failed
        """
        try:
            # Ensure we have enough data
            if len(data) < sequence_length:
                logger.error(f"Insufficient data: need {sequence_length}, got {len(data)}")
                return None
            
            # Select required features
            available_features = [f for f in features if f in data.columns]
            if not available_features:
                logger.error(f"No required features found in data: {features}")
                return None
            
            # Get the last sequence_length data points
            sequence_data = data[available_features].tail(sequence_length)
            
            # Scale the data
            scaled_data = scaler.transform(sequence_data.values)
            
            # Reshape for LSTM input (batch_size, sequence_length, features)
            sequence = scaled_data.reshape(1, sequence_length, len(available_features))
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error creating prediction sequence: {e}")
            return None
    
    def update_prediction_sequence(self, current_sequence: np.ndarray, 
                                 new_price: float, scaler, features: List[str]) -> np.ndarray:
        """
        Update prediction sequence for next prediction (rolling window)
        
        Args:
            current_sequence: Current input sequence
            new_price: New predicted price
            scaler: Fitted scaler
            features: List of feature names
            
        Returns:
            Updated sequence
        """
        try:
            # Shift the sequence by removing the first time step
            updated_sequence = current_sequence.copy()
            updated_sequence[0, :-1, :] = current_sequence[0, 1:, :]
            
            # Create new feature row based on predicted price
            # This is a simplified approach - in production you'd want to
            # calculate other features (volume, indicators, etc.) based on the new price
            new_features = []
            for feature in features:
                if feature == 'close':
                    new_features.append(new_price)
                elif feature == 'open':
                    # Use previous close as open (simplified)
                    new_features.append(new_price)
                elif feature == 'high':
                    # Assume high is slightly above close
                    new_features.append(new_price * 1.001)
                elif feature == 'low':
                    # Assume low is slightly below close
                    new_features.append(new_price * 0.999)
                elif feature == 'volume':
                    # Use average volume from recent data
                    recent_volumes = current_sequence[0, :, features.index('volume') if 'volume' in features else 0]
                    new_features.append(np.mean(recent_volumes))
                else:
                    # For technical indicators, use previous values or calculate
                    # This is a simplified approach - in production you'd calculate proper values
                    prev_values = current_sequence[0, -1, features.index(feature)]
                    new_features.append(prev_values)
            
            # Scale the new features
            new_scaled_features = scaler.transform([new_features])[0]
            updated_sequence[0, -1, :] = new_scaled_features
            
            return updated_sequence
            
        except Exception as e:
            logger.error(f"Error updating prediction sequence: {e}")
            return current_sequence
    
    def validate_prediction_output(self, prediction: float) -> bool:
        """
        Validate prediction output
        
        Args:
            prediction: Predicted price
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if prediction is not NaN or infinite
            if np.isnan(prediction) or np.isinf(prediction):
                return False
            
            # Check if prediction is positive
            if prediction <= 0:
                return False
            
            # Check if prediction is within reasonable bounds
            if prediction > 10000 or prediction < 0.01:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating prediction: {e}")
            return False
    
    def calculate_prediction_confidence(self, symbol: str, predictions: List[float], 
                                      historical_data: pd.DataFrame) -> float:
        """
        Calculate confidence score for predictions
        
        Args:
            symbol: Stock symbol
            predictions: List of predicted prices
            historical_data: Historical market data
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Get recent model performance
            recent_predictions = self.data_storage.retrieve_historical_data(
                symbol, 
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
            
            if recent_predictions.empty:
                # If no recent data, use a default confidence
                return 0.5
            
            # Calculate volatility-based confidence
            recent_returns = historical_data['close'].pct_change().dropna()
            volatility = recent_returns.std()
            
            # Lower volatility = higher confidence
            volatility_confidence = max(0.1, 1.0 - (volatility * 10))
            
            # Calculate trend consistency
            recent_trend = historical_data['close'].tail(10).pct_change().mean()
            trend_confidence = 1.0 - abs(recent_trend) * 5
            
            # Combine confidence factors
            confidence = (volatility_confidence + trend_confidence) / 2
            confidence = max(0.1, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence for {symbol}: {e}")
            return 0.5  # Default confidence
    
    def store_predictions_with_decay(self, symbol: str, predictions_df: pd.DataFrame, 
                                   base_confidence: float) -> bool:
        """
        Store predictions in database with time-based confidence decay
        
        Args:
            symbol: Stock symbol
            predictions_df: DataFrame with predictions
            base_confidence: Base confidence score for first prediction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prediction_timestamp = datetime.now()
            
            for i, (timestamp, row) in enumerate(predictions_df.iterrows()):
                # Calculate confidence decay: closer predictions have higher confidence
                # Decay factor: 0.95 for day 1, 0.80 for day 7, 0.65 for day 14
                days_ahead = i + 1
                decay_factor = max(0.3, 1.0 - (days_ahead - 1) * 0.025)  # 2.5% decay per day
                individual_confidence = base_confidence * decay_factor
                
                self.data_storage.store_prediction(
                    symbol=symbol,
                    prediction_timestamp=prediction_timestamp,
                    prediction_date=timestamp,
                    predicted_price=row['predicted_price'],
                    confidence_score=individual_confidence
                )
            
            logger.info(f"Stored {len(predictions_df)} predictions for {symbol} with confidence decay")
            return True
            
        except Exception as e:
            logger.error(f"Error storing predictions for {symbol}: {e}")
            return False
    
    def store_predictions(self, symbol: str, predictions_df: pd.DataFrame, 
                         confidence_score: float) -> bool:
        """
        Store predictions in database (legacy method)
        
        Args:
            symbol: Stock symbol
            predictions_df: DataFrame with predictions
            confidence_score: Overall confidence score
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prediction_timestamp = datetime.now()
            
            for timestamp, row in predictions_df.iterrows():
                self.data_storage.store_prediction(
                    symbol=symbol,
                    prediction_timestamp=prediction_timestamp,
                    prediction_date=timestamp,
                    predicted_price=row['predicted_price'],
                    confidence_score=confidence_score
                )
            
            logger.info(f"Stored {len(predictions_df)} predictions for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing predictions for {symbol}: {e}")
            return False
    
    def generate_trade_recommendations(self, symbol: str, predictions_df: pd.DataFrame, 
                                     confidence_score: float, max_recommendations: int = 1) -> List[Dict[str, Any]]:
        """
        Generate trade recommendations based on predictions - selects highest profiting recommendations
        
        Args:
            symbol: Stock symbol
            predictions_df: DataFrame with predictions
            confidence_score: Overall confidence score
            max_recommendations: Maximum number of recommendations to return (default: 1 for highest profit)
            
        Returns:
            List of trade recommendations (sorted by profit, highest first)
        """
        try:
            all_recommendations = []
            predictions = predictions_df['predicted_price'].values
            timestamps = predictions_df.index
            
            # Find all potential entry and exit points
            for i in range(len(predictions) - 1):
                entry_price = predictions[i]
                entry_time = timestamps[i]
                
                # Look for profitable exit points
                for j in range(i + 1, min(i + 100, len(predictions))):  # Look ahead up to 100 predictions
                    exit_price = predictions[j]
                    exit_time = timestamps[j]
                    
                    profit_percent = ((exit_price - entry_price) / entry_price) * 100
                    
                    # Only consider positive profit opportunities
                    if profit_percent > 0:
                        recommendation = {
                            'symbol': symbol,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'expected_profit_percent': profit_percent,
                            'confidence_score': confidence_score,
                            'duration_hours': (exit_time - entry_time).total_seconds() / 3600
                        }
                        
                        all_recommendations.append(recommendation)
            
            # Sort by profit percentage (highest first)
            all_recommendations.sort(key=lambda x: x['expected_profit_percent'], reverse=True)
            
            # Select top recommendations (non-overlapping)
            selected_recommendations = []
            for recommendation in all_recommendations:
                if len(selected_recommendations) >= max_recommendations:
                    break
                    
                # Check if this recommendation doesn't overlap with selected ones
                if not self.recommendation_overlaps(selected_recommendations, 
                                                   recommendation['entry_time'], 
                                                   recommendation['exit_time']):
                    selected_recommendations.append(recommendation)
                    
                    # Store in database - convert pandas Timestamps to datetime
                    self.data_storage.store_trade_recommendation(
                        symbol=symbol,
                        entry_time=recommendation['entry_time'].to_pydatetime() if hasattr(recommendation['entry_time'], 'to_pydatetime') else recommendation['entry_time'],
                        exit_time=recommendation['exit_time'].to_pydatetime() if hasattr(recommendation['exit_time'], 'to_pydatetime') else recommendation['exit_time'],
                        entry_price=recommendation['entry_price'],
                        exit_price=recommendation['exit_price'],
                        confidence_score=confidence_score
                    )
            
            # Calculate overall statistics for debugging
            if len(predictions) > 1:
                all_profits = []
                
                for i in range(len(predictions) - 1):
                    for j in range(i + 1, min(i + 100, len(predictions))):
                        profit_percent = ((predictions[j] - predictions[i]) / predictions[i]) * 100
                        all_profits.append(profit_percent)
                
                # Calculate profit distribution
                if all_profits:
                    max_profit = max(all_profits)
                    avg_profit = sum(all_profits) / len(all_profits)
                    positive_count = sum(1 for p in all_profits if p > 0)
                    negative_count = sum(1 for p in all_profits if p < 0)
                    
                    logger.info(f"Trade recommendation analysis for {symbol}:")
                    logger.info(f"  - Total predictions analyzed: {len(predictions)}")
                    logger.info(f"  - Total opportunities evaluated: {len(all_profits)}")
                    logger.info(f"  - Maximum profit opportunity: {max_profit:.2f}%")
                    logger.info(f"  - Average profit opportunity: {avg_profit:.2f}%")
                    logger.info(f"  - Positive opportunities: {positive_count}")
                    logger.info(f"  - Negative opportunities: {negative_count}")
                    logger.info(f"  - Generated recommendations: {len(selected_recommendations)}")
                    
                    # Show top profit opportunities
                    if selected_recommendations:
                        top_profits = [r['expected_profit_percent'] for r in selected_recommendations]
                        formatted_profits = [f"{p:.2f}%" for p in top_profits]
                        logger.info(f"  - Selected recommendations (profit %): {formatted_profits}")
            
            logger.info(f"Generated {len(selected_recommendations)} trade recommendations for {symbol}")
            return selected_recommendations
            
        except Exception as e:
            logger.error(f"Error generating trade recommendations for {symbol}: {e}")
            return []
    
    def recommendation_overlaps(self, existing_recommendations: List[Dict[str, Any]], 
                              entry_time: datetime, exit_time: datetime) -> bool:
        """
        Check if a new recommendation overlaps with existing ones
        
        Args:
            existing_recommendations: List of existing recommendations
            entry_time: New recommendation entry time
            exit_time: New recommendation exit time
            
        Returns:
            True if overlaps, False otherwise
        """
        for rec in existing_recommendations:
            existing_entry = rec['entry_time']
            existing_exit = rec['exit_time']
            
            # Check for overlap
            if (entry_time < existing_exit and exit_time > existing_entry):
                return True
        
        return False
    
    def get_prediction_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of latest predictions for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with prediction summary
        """
        try:
            # Get latest predictions from database
            latest_data = self.data_storage.get_latest_data(symbol, hours_back=1)
            if latest_data.empty:
                return None
            
            current_price = latest_data['close'].iloc[-1]
            
            # Get active recommendations
            active_recommendations = self.data_storage.get_active_recommendations(symbol)
            
            summary = {
                'symbol': symbol,
                'current_price': current_price,
                'last_updated': latest_data.index[-1],
                'active_recommendations': len(active_recommendations),
                'recommendations': active_recommendations.to_dict('records') if not active_recommendations.empty else []
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting prediction summary for {symbol}: {e}")
            return None
    
    def _get_next_trading_day(self, date: datetime) -> datetime:
        """
        Get the next trading day (Monday-Friday, excluding major holidays)
        
        Args:
            date: Starting date
            
        Returns:
            Next trading day at 4 PM (market close)
        """
        # Major US market holidays (simplified list)
        holidays = {
            (1, 1): "New Year's Day",
            (7, 4): "Independence Day", 
            (12, 25): "Christmas Day",
            (11, 28): "Thanksgiving",  # 4th Thursday of November (simplified)
            (1, 15): "MLK Day",  # 3rd Monday of January (simplified)
            (2, 19): "Presidents Day",  # 3rd Monday of February (simplified)
            (5, 27): "Memorial Day",  # Last Monday of May (simplified)
            (9, 2): "Labor Day",  # 1st Monday of September (simplified)
        }
        
        # Skip weekends and holidays
        while True:
            # Check if it's a weekend (Saturday=5, Sunday=6)
            if date.weekday() >= 5:
                date += timedelta(days=1)
                continue
            
            # Check if it's a holiday
            month_day = (date.month, date.day)
            if month_day in holidays:
                logger.debug(f"Skipping holiday: {holidays[month_day]} on {date.date()}")
                date += timedelta(days=1)
                continue
            
            # Found a trading day
            break
        
        # Set to 4 PM (market close)
        return date.replace(hour=16, minute=0, second=0, microsecond=0)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # This would be used with actual ModelManager and DataStorage instances
    print("PredictionEngine module loaded successfully")
    print("Use with ModelManager and DataStorage instances for full functionality")
