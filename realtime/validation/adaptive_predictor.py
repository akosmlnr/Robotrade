"""
Adaptive Prediction System for Real-time LSTM Prediction System
Phase 2.1: Advanced validation and adaptive prediction with reprediction logic
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status enumeration"""
    VALID = "valid"
    NEEDS_REPREDICTION = "needs_reprediction"
    INVALID = "invalid"
    UNKNOWN = "unknown"

@dataclass
class ValidationResult:
    """Result of prediction validation"""
    status: ValidationStatus
    error_percentage: float
    confidence_score: float
    validation_timestamp: datetime
    actual_price: Optional[float] = None
    predicted_price: Optional[float] = None
    market_volatility: Optional[float] = None
    recommendation: Optional[str] = None

class AdaptivePredictor:
    """
    Advanced adaptive prediction system with validation and reprediction logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adaptive predictor
        
        Args:
            config: Configuration dictionary with validation thresholds
        """
        self.config = config
        self.validation_threshold = config.get('validation_threshold', 0.02)  # 2%
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.volatility_threshold = config.get('volatility_threshold', 0.05)  # 5%
        self.max_repredictions = config.get('max_repredictions', 3)
        self.validation_window = config.get('validation_window_hours', 24)
        
        # Performance tracking
        self.validation_history = {}
        self.reprediction_counts = {}
        self.performance_metrics = {}
        
        logger.info(f"AdaptivePredictor initialized with validation threshold: {self.validation_threshold}")
    
    def validate_prediction(self, symbol: str, predicted_price: float, 
                          actual_price: float, prediction_timestamp: datetime,
                          market_data: pd.DataFrame) -> ValidationResult:
        """
        Validate a prediction against actual market data
        
        Args:
            symbol: Stock symbol
            predicted_price: Predicted price
            actual_price: Actual market price
            prediction_timestamp: When the prediction was made
            market_data: Recent market data for context
            
        Returns:
            ValidationResult with validation status and metrics
        """
        try:
            # Calculate prediction error
            error_percentage = abs(predicted_price - actual_price) / actual_price
            
            # Calculate market volatility
            market_volatility = self._calculate_market_volatility(market_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_validation_confidence(
                error_percentage, market_volatility, symbol
            )
            
            # Determine validation status
            status = self._determine_validation_status(
                error_percentage, confidence_score, market_volatility
            )
            
            # Generate recommendation
            recommendation = self._generate_validation_recommendation(
                status, error_percentage, confidence_score
            )
            
            result = ValidationResult(
                status=status,
                error_percentage=error_percentage,
                confidence_score=confidence_score,
                validation_timestamp=datetime.now(),
                actual_price=actual_price,
                predicted_price=predicted_price,
                market_volatility=market_volatility,
                recommendation=recommendation
            )
            
            # Store validation history
            self._store_validation_result(symbol, result)
            
            logger.info(f"Validation for {symbol}: {status.value}, error: {error_percentage:.2%}, confidence: {confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating prediction for {symbol}: {e}")
            return ValidationResult(
                status=ValidationStatus.UNKNOWN,
                error_percentage=0.0,
                confidence_score=0.0,
                validation_timestamp=datetime.now(),
                recommendation="Validation failed"
            )
    
    def should_repredict(self, symbol: str, validation_result: ValidationResult) -> bool:
        """
        Determine if a reprediction is needed based on validation results
        
        Args:
            symbol: Stock symbol
            validation_result: Result of prediction validation
            
        Returns:
            True if reprediction is needed, False otherwise
        """
        try:
            # Check if we've exceeded max repredictions
            current_count = self.reprediction_counts.get(symbol, 0)
            if current_count >= self.max_repredictions:
                logger.warning(f"Max repredictions reached for {symbol}: {current_count}")
                return False
            
            # Check validation status
            if validation_result.status == ValidationStatus.NEEDS_REPREDICTION:
                return True
            
            # Check error threshold
            if validation_result.error_percentage > self.validation_threshold:
                return True
            
            # Check confidence threshold
            if validation_result.confidence_score < self.confidence_threshold:
                return True
            
            # Check for high volatility
            if validation_result.market_volatility and validation_result.market_volatility > self.volatility_threshold:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining reprediction need for {symbol}: {e}")
            return False
    
    def trigger_reprediction(self, symbol: str, reason: str) -> bool:
        """
        Trigger a reprediction for a symbol
        
        Args:
            symbol: Stock symbol
            reason: Reason for reprediction
            
        Returns:
            True if reprediction was triggered, False otherwise
        """
        try:
            # Increment reprediction count
            self.reprediction_counts[symbol] = self.reprediction_counts.get(symbol, 0) + 1
            
            logger.info(f"Triggering reprediction for {symbol} (reason: {reason}, count: {self.reprediction_counts[symbol]})")
            
            # Reset reprediction count after successful validation
            # This will be called by the main system after successful reprediction
            
            return True
            
        except Exception as e:
            logger.error(f"Error triggering reprediction for {symbol}: {e}")
            return False
    
    def reset_reprediction_count(self, symbol: str):
        """Reset reprediction count for a symbol after successful prediction"""
        self.reprediction_counts[symbol] = 0
        logger.debug(f"Reset reprediction count for {symbol}")
    
    def _calculate_market_volatility(self, market_data: pd.DataFrame) -> float:
        """
        Calculate market volatility from recent data
        
        Args:
            market_data: Recent market data
            
        Returns:
            Volatility as standard deviation of returns
        """
        try:
            if len(market_data) < 2:
                return 0.0
            
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Calculate volatility (standard deviation of returns)
            volatility = returns.std()
            
            return float(volatility) if not np.isnan(volatility) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating market volatility: {e}")
            return 0.0
    
    def _calculate_validation_confidence(self, error_percentage: float, 
                                       market_volatility: float, symbol: str) -> float:
        """
        Calculate confidence score for validation
        
        Args:
            error_percentage: Prediction error percentage
            market_volatility: Market volatility
            symbol: Stock symbol
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Base confidence from error percentage (lower error = higher confidence)
            error_confidence = max(0.0, 1.0 - (error_percentage / self.validation_threshold))
            
            # Volatility factor (lower volatility = higher confidence)
            volatility_factor = max(0.1, 1.0 - (market_volatility / self.volatility_threshold))
            
            # Historical performance factor
            historical_factor = self._get_historical_performance_factor(symbol)
            
            # Combine factors
            confidence = (error_confidence * 0.4 + volatility_factor * 0.3 + historical_factor * 0.3)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating validation confidence: {e}")
            return 0.5
    
    def _get_historical_performance_factor(self, symbol: str) -> float:
        """
        Get historical performance factor for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Performance factor between 0 and 1
        """
        try:
            if symbol not in self.validation_history:
                return 0.5  # Default for new symbols
            
            recent_validations = self.validation_history[symbol][-10:]  # Last 10 validations
            
            if not recent_validations:
                return 0.5
            
            # Calculate average confidence from recent validations
            avg_confidence = np.mean([v.confidence_score for v in recent_validations])
            
            return float(avg_confidence) if not np.isnan(avg_confidence) else 0.5
            
        except Exception as e:
            logger.error(f"Error getting historical performance factor for {symbol}: {e}")
            return 0.5
    
    def _determine_validation_status(self, error_percentage: float, 
                                   confidence_score: float, market_volatility: float) -> ValidationStatus:
        """
        Determine validation status based on metrics
        
        Args:
            error_percentage: Prediction error percentage
            confidence_score: Confidence score
            market_volatility: Market volatility
            
        Returns:
            ValidationStatus
        """
        try:
            # High error and low confidence = needs reprediction
            if error_percentage > self.validation_threshold and confidence_score < self.confidence_threshold:
                return ValidationStatus.NEEDS_REPREDICTION
            
            # Very high error = invalid
            if error_percentage > (self.validation_threshold * 2):
                return ValidationStatus.INVALID
            
            # High volatility with moderate error = needs reprediction
            if market_volatility > self.volatility_threshold and error_percentage > (self.validation_threshold * 0.5):
                return ValidationStatus.NEEDS_REPREDICTION
            
            # Low error and high confidence = valid
            if error_percentage <= self.validation_threshold and confidence_score >= self.confidence_threshold:
                return ValidationStatus.VALID
            
            # Default to needs reprediction for uncertain cases
            return ValidationStatus.NEEDS_REPREDICTION
            
        except Exception as e:
            logger.error(f"Error determining validation status: {e}")
            return ValidationStatus.UNKNOWN
    
    def _generate_validation_recommendation(self, status: ValidationStatus, 
                                          error_percentage: float, confidence_score: float) -> str:
        """
        Generate recommendation based on validation results
        
        Args:
            status: Validation status
            error_percentage: Prediction error percentage
            confidence_score: Confidence score
            
        Returns:
            Recommendation string
        """
        try:
            if status == ValidationStatus.VALID:
                return f"Prediction is accurate (error: {error_percentage:.2%}, confidence: {confidence_score:.2f})"
            
            elif status == ValidationStatus.NEEDS_REPREDICTION:
                return f"Reprediction recommended (error: {error_percentage:.2%}, confidence: {confidence_score:.2f})"
            
            elif status == ValidationStatus.INVALID:
                return f"Prediction is invalid (error: {error_percentage:.2%}, confidence: {confidence_score:.2f})"
            
            else:
                return "Validation status unknown"
                
        except Exception as e:
            logger.error(f"Error generating validation recommendation: {e}")
            return "Recommendation generation failed"
    
    def _store_validation_result(self, symbol: str, result: ValidationResult):
        """
        Store validation result in history
        
        Args:
            symbol: Stock symbol
            result: Validation result
        """
        try:
            if symbol not in self.validation_history:
                self.validation_history[symbol] = []
            
            self.validation_history[symbol].append(result)
            
            # Keep only recent validations (last 100)
            if len(self.validation_history[symbol]) > 100:
                self.validation_history[symbol] = self.validation_history[symbol][-100:]
            
        except Exception as e:
            logger.error(f"Error storing validation result for {symbol}: {e}")
    
    def get_validation_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get validation summary for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with validation summary
        """
        try:
            if symbol not in self.validation_history:
                return {
                    'symbol': symbol,
                    'total_validations': 0,
                    'avg_error_percentage': 0.0,
                    'avg_confidence_score': 0.0,
                    'validation_status_distribution': {},
                    'reprediction_count': 0
                }
            
            validations = self.validation_history[symbol]
            
            # Calculate metrics
            total_validations = len(validations)
            avg_error = np.mean([v.error_percentage for v in validations])
            avg_confidence = np.mean([v.confidence_score for v in validations])
            
            # Status distribution
            status_dist = {}
            for validation in validations:
                status = validation.status.value
                status_dist[status] = status_dist.get(status, 0) + 1
            
            return {
                'symbol': symbol,
                'total_validations': total_validations,
                'avg_error_percentage': float(avg_error) if not np.isnan(avg_error) else 0.0,
                'avg_confidence_score': float(avg_confidence) if not np.isnan(avg_confidence) else 0.0,
                'validation_status_distribution': status_dist,
                'reprediction_count': self.reprediction_counts.get(symbol, 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting validation summary for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """
        Get overall system performance metrics
        
        Returns:
            Dictionary with system performance metrics
        """
        try:
            all_validations = []
            for symbol_validations in self.validation_history.values():
                all_validations.extend(symbol_validations)
            
            if not all_validations:
                return {
                    'total_validations': 0,
                    'avg_error_percentage': 0.0,
                    'avg_confidence_score': 0.0,
                    'total_repredictions': 0,
                    'symbols_monitored': 0
                }
            
            # Calculate overall metrics
            total_validations = len(all_validations)
            avg_error = np.mean([v.error_percentage for v in all_validations])
            avg_confidence = np.mean([v.confidence_score for v in all_validations])
            total_repredictions = sum(self.reprediction_counts.values())
            symbols_monitored = len(self.validation_history)
            
            return {
                'total_validations': total_validations,
                'avg_error_percentage': float(avg_error) if not np.isnan(avg_error) else 0.0,
                'avg_confidence_score': float(avg_confidence) if not np.isnan(avg_confidence) else 0.0,
                'total_repredictions': total_repredictions,
                'symbols_monitored': symbols_monitored,
                'validation_threshold': self.validation_threshold,
                'confidence_threshold': self.confidence_threshold,
                'volatility_threshold': self.volatility_threshold
            }
            
        except Exception as e:
            logger.error(f"Error getting system performance metrics: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = {
        'validation_threshold': 0.02,
        'confidence_threshold': 0.6,
        'volatility_threshold': 0.05,
        'max_repredictions': 3,
        'validation_window_hours': 24
    }
    
    # Test the adaptive predictor
    predictor = AdaptivePredictor(config)
    
    # Create sample market data
    sample_data = pd.DataFrame({
        'close': [100, 101, 102, 101, 103, 102, 104]
    })
    
    # Test validation
    result = predictor.validate_prediction(
        symbol='AAPL',
        predicted_price=105.0,
        actual_price=103.0,
        prediction_timestamp=datetime.now(),
        market_data=sample_data
    )
    
    print(f"Validation result: {result.status.value}")
    print(f"Error percentage: {result.error_percentage:.2%}")
    print(f"Confidence score: {result.confidence_score:.2f}")
    print(f"Recommendation: {result.recommendation}")
    
    # Test reprediction logic
    should_repredict = predictor.should_repredict('AAPL', result)
    print(f"Should repredict: {should_repredict}")
    
    print("Adaptive predictor test completed")
