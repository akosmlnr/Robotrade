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
            features = config['features']
            
            # Get latest market data - need enough for sequence_length (25) plus some buffer
            # Since we need 25 data points and get 15-minute intervals, we need at least 25 * 15 minutes = 375 minutes = 6.25 hours
            # But since we're on weekend, let's look back further to get enough data
            hours_back = max(24, sequence_length * 15 // 60 + 24)  # At least 24 hours, or enough for sequence + 24 hour buffer
            latest_data = self.data_storage.get_latest_data(symbol, hours_back=hours_back)
            
            if latest_data.empty:
                logger.error(f"No market data available for {symbol}")
                return None
            
            logger.info(f"Retrieved {len(latest_data)} records for {symbol} prediction (need {sequence_length})")
            
            # Create prediction sequence
            prediction_sequence = self.create_prediction_sequence(
                latest_data, sequence_length, features, scaler
            )
            
            if prediction_sequence is None:
                logger.error(f"Failed to create prediction sequence for {symbol}")
                return None
            
            # Generate 2-week prediction (14 days * 24 hours * 4 = 1344 15-minute intervals)
            biweekly_intervals = 14 * 24 * 4
            predictions = []
            timestamps = []
            
            # Start from the next 15-minute interval
            current_time = latest_data.index[-1] + timedelta(minutes=15)
            
            # Use the last known price as starting point
            current_price = latest_data['close'].iloc[-1]
            logger.info(f"Starting prediction from price: ${current_price:.2f} at {current_time}")
            
            # Safety check for valid current price
            if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
                logger.error(f"Invalid current price for {symbol}: {current_price}")
                return None
            
            # Generate predictions step by step
            for i in range(biweekly_intervals):
                try:
                    # Make prediction for next time step
                    prediction_scaled = model.predict(prediction_sequence, verbose=0)
                    prediction_price = scaler.inverse_transform(prediction_scaled)[0][0]
                    
                    # Debug logging for first few predictions
                    if i < 5:
                        logger.info(f"Step {i}: Scaled prediction: {prediction_scaled[0][0]:.6f}, "
                                   f"Inverse transform: {prediction_price:.2f}, "
                                   f"Previous price: {current_price:.2f}")
                    
                    # Validate prediction
                    if not self.validate_prediction_output(prediction_price):
                        logger.warning(f"Invalid prediction for {symbol} at step {i}")
                        # Use previous prediction or current price
                        prediction_price = predictions[-1] if predictions else current_price
                    
                    predictions.append(prediction_price)
                    timestamps.append(current_time)
                    
                    # Update sequence for next prediction (rolling window)
                    prediction_sequence = self.update_prediction_sequence(
                        prediction_sequence, prediction_price, scaler, features
                    )
                    
                    # Move to next time step
                    current_time += timedelta(minutes=15)
                    
                except Exception as e:
                    logger.error(f"Error in prediction step {i} for {symbol}: {e}")
                    # Use previous prediction or current price as fallback
                    fallback_price = predictions[-1] if predictions else current_price
                    predictions.append(fallback_price)
                    timestamps.append(current_time)
                    current_time += timedelta(minutes=15)
            
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
            
            # Store predictions in database
            self.store_predictions(symbol, prediction_df, confidence_score)
            
            result = {
                'symbol': symbol,
                'prediction_timestamp': datetime.now(),
                'predictions': prediction_df,
                'confidence_score': confidence_score,
                'sequence_length': sequence_length,
                'total_predictions': len(predictions),
                'prediction_horizon': '2 weeks',
                'data_points': biweekly_intervals
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
    
    def store_predictions(self, symbol: str, predictions_df: pd.DataFrame, 
                         confidence_score: float) -> bool:
        """
        Store predictions in database
        
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
                                     confidence_score: float, min_profit_percent: float = 2.0) -> List[Dict[str, Any]]:
        """
        Generate trade recommendations based on predictions
        
        Args:
            symbol: Stock symbol
            predictions_df: DataFrame with predictions
            confidence_score: Overall confidence score
            min_profit_percent: Minimum profit percentage for recommendations
            
        Returns:
            List of trade recommendations
        """
        try:
            recommendations = []
            predictions = predictions_df['predicted_price'].values
            timestamps = predictions_df.index
            
            # Find potential entry and exit points
            current_price = predictions[0]
            
            for i in range(len(predictions) - 1):
                entry_price = predictions[i]
                entry_time = timestamps[i]
                
                # Look for profitable exit points
                for j in range(i + 1, min(i + 100, len(predictions))):  # Look ahead up to 25 hours
                    exit_price = predictions[j]
                    exit_time = timestamps[j]
                    
                    profit_percent = ((exit_price - entry_price) / entry_price) * 100
                    
                    # Debug logging for first few iterations
                    if i < 3 and j < i + 10:
                        logger.debug(f"Checking profit: entry=${entry_price:.2f}, exit=${exit_price:.2f}, profit={profit_percent:.2f}%")
                    
                    if profit_percent >= min_profit_percent:
                        # Check if this recommendation doesn't overlap with existing ones
                        if not self.recommendation_overlaps(recommendations, entry_time, exit_time):
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
                            
                            recommendations.append(recommendation)
                            
                            # Store in database
                            self.data_storage.store_trade_recommendation(
                                symbol=symbol,
                                entry_time=entry_time,
                                exit_time=exit_time,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                confidence_score=confidence_score
                            )
                            
                            # Skip ahead to avoid overlapping recommendations
                            i = j
                            break
            
            # Calculate overall statistics for debugging
            if len(predictions) > 1:
                max_profit = 0
                profitable_count = 0
                all_profits = []
                
                for i in range(len(predictions) - 1):
                    for j in range(i + 1, min(i + 100, len(predictions))):
                        profit_percent = ((predictions[j] - predictions[i]) / predictions[i]) * 100
                        all_profits.append(profit_percent)
                        max_profit = max(max_profit, profit_percent)
                        if profit_percent >= min_profit_percent:
                            profitable_count += 1
                
                # Calculate profit distribution
                if all_profits:
                    avg_profit = sum(all_profits) / len(all_profits)
                    positive_count = sum(1 for p in all_profits if p > 0)
                    under_2_percent = sum(1 for p in all_profits if 0 < p < min_profit_percent)
                    negative_count = sum(1 for p in all_profits if p < 0)
                    
                    logger.info(f"Trade recommendation analysis for {symbol}:")
                    logger.info(f"  - Total predictions analyzed: {len(predictions)}")
                    logger.info(f"  - Total opportunities evaluated: {len(all_profits)}")
                    logger.info(f"  - Maximum profit opportunity: {max_profit:.2f}%")
                    logger.info(f"  - Average profit opportunity: {avg_profit:.2f}%")
                    logger.info(f"  - Positive opportunities: {positive_count}")
                    logger.info(f"  - Opportunities under {min_profit_percent}%: {under_2_percent}")
                    logger.info(f"  - Opportunities >= {min_profit_percent}%: {profitable_count}")
                    logger.info(f"  - Negative opportunities: {negative_count}")
                    logger.info(f"  - Generated recommendations: {len(recommendations)}")
                    
                    # Show some examples of under-2% opportunities
                    if under_2_percent > 0:
                        under_2_examples = [p for p in all_profits if 0 < p < min_profit_percent]
                        under_2_examples.sort(reverse=True)
                        logger.info(f"  - Best opportunities under {min_profit_percent}%: {under_2_examples[:5]}")
            
            logger.info(f"Generated {len(recommendations)} trade recommendations for {symbol}")
            return recommendations
            
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


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # This would be used with actual ModelManager and DataStorage instances
    print("PredictionEngine module loaded successfully")
    print("Use with ModelManager and DataStorage instances for full functionality")
