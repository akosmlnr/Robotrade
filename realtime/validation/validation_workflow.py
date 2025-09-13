"""
Prediction Validation Workflow for Real-time LSTM Prediction System
Phase 3.1: Comprehensive prediction validation with multiple validation layers
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status levels"""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    CRITICAL = "critical"

class ValidationType(Enum):
    """Types of validation checks"""
    PRICE_RANGE = "price_range"
    VOLATILITY = "volatility"
    TREND_CONSISTENCY = "trend_consistency"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    HISTORICAL_ACCURACY = "historical_accuracy"
    MARKET_CONDITIONS = "market_conditions"
    MODEL_PERFORMANCE = "model_performance"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    validation_type: ValidationType
    status: ValidationStatus
    score: float  # 0-1, where 1 is perfect
    message: str
    details: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class OverallValidationResult:
    """Overall validation result for a prediction"""
    symbol: str
    prediction_timestamp: datetime
    overall_status: ValidationStatus
    overall_score: float
    validation_results: List[ValidationResult]
    recommendations: List[str]
    requires_reprediction: bool = False
    confidence_adjustment: float = 0.0

class PredictionValidationWorkflow:
    """
    Comprehensive prediction validation workflow with multiple validation layers
    """
    
    def __init__(self, data_storage, model_manager):
        """
        Initialize the validation workflow
        
        Args:
            data_storage: DataStorage instance
            model_manager: ModelManager instance
        """
        self.data_storage = data_storage
        self.model_manager = model_manager
        
        # Validation thresholds
        self.price_change_threshold = 0.15  # 15% max change in 15 minutes
        self.volatility_threshold = 0.05    # 5% volatility threshold
        self.confidence_threshold = 0.3     # Minimum confidence score
        self.trend_consistency_threshold = 0.7  # Trend consistency threshold
        self.historical_accuracy_threshold = 0.6  # Historical accuracy threshold
        
        # Market condition thresholds
        self.market_volatility_threshold = 0.08  # Market-wide volatility threshold
        self.volume_anomaly_threshold = 3.0      # Volume anomaly threshold (3x normal)
        
        # Validation weights
        self.validation_weights = {
            ValidationType.PRICE_RANGE: 0.25,
            ValidationType.VOLATILITY: 0.20,
            ValidationType.TREND_CONSISTENCY: 0.15,
            ValidationType.CONFIDENCE_THRESHOLD: 0.15,
            ValidationType.HISTORICAL_ACCURACY: 0.15,
            ValidationType.MARKET_CONDITIONS: 0.10
        }
        
        logger.info("PredictionValidationWorkflow initialized")
    
    def validate_prediction(self, symbol: str, prediction_result: Dict[str, Any]) -> OverallValidationResult:
        """
        Validate a prediction result using multiple validation layers
        
        Args:
            symbol: Stock symbol
            prediction_result: Prediction result from PredictionEngine
            
        Returns:
            OverallValidationResult with comprehensive validation
        """
        try:
            validation_results = []
            
            # Run all validation checks
            validation_results.append(self._validate_price_range(symbol, prediction_result))
            validation_results.append(self._validate_volatility(symbol, prediction_result))
            validation_results.append(self._validate_trend_consistency(symbol, prediction_result))
            validation_results.append(self._validate_confidence_threshold(symbol, prediction_result))
            validation_results.append(self._validate_historical_accuracy(symbol, prediction_result))
            validation_results.append(self._validate_market_conditions(symbol, prediction_result))
            
            # Calculate overall validation result
            overall_result = self._calculate_overall_validation(
                symbol, prediction_result, validation_results
            )
            
            # Store validation results
            self._store_validation_results(overall_result)
            
            logger.info(f"Validation completed for {symbol}: {overall_result.overall_status.value}")
            return overall_result
            
        except Exception as e:
            logger.error(f"Error validating prediction for {symbol}: {e}")
            return self._create_error_validation_result(symbol, str(e))
    
    def _validate_price_range(self, symbol: str, prediction_result: Dict[str, Any]) -> ValidationResult:
        """Validate price range and changes"""
        try:
            predictions_df = prediction_result['predictions']
            predictions = predictions_df['predicted_price'].values
            
            # Get current price
            current_data = self.data_storage.get_latest_data(symbol, hours_back=1)
            if current_data.empty:
                return ValidationResult(
                    ValidationType.PRICE_RANGE,
                    ValidationStatus.INVALID,
                    0.0,
                    "No current price data available",
                    {}
                )
            
            current_price = current_data['close'].iloc[-1]
            
            # Check for extreme price changes
            max_change = 0.0
            for pred_price in predictions:
                change = abs(pred_price - current_price) / current_price
                max_change = max(max_change, change)
            
            # Calculate score based on price change
            if max_change <= self.price_change_threshold * 0.5:
                status = ValidationStatus.VALID
                score = 1.0
                message = f"Price changes within normal range ({max_change:.2%})"
            elif max_change <= self.price_change_threshold:
                status = ValidationStatus.WARNING
                score = 0.7
                message = f"Price changes approaching threshold ({max_change:.2%})"
            else:
                status = ValidationStatus.INVALID
                score = 0.0
                message = f"Price changes exceed threshold ({max_change:.2%})"
            
            # Check for negative prices
            if any(p <= 0 for p in predictions):
                status = ValidationStatus.CRITICAL
                score = 0.0
                message = "Negative or zero prices detected"
            
            return ValidationResult(
                ValidationType.PRICE_RANGE,
                status,
                score,
                message,
                {
                    'current_price': current_price,
                    'max_price_change': max_change,
                    'min_predicted_price': min(predictions),
                    'max_predicted_price': max(predictions)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in price range validation: {e}")
            return ValidationResult(
                ValidationType.PRICE_RANGE,
                ValidationStatus.INVALID,
                0.0,
                f"Price range validation failed: {e}",
                {}
            )
    
    def _validate_volatility(self, symbol: str, prediction_result: Dict[str, Any]) -> ValidationResult:
        """Validate prediction volatility"""
        try:
            predictions_df = prediction_result['predictions']
            predictions = predictions_df['predicted_price'].values
            
            # Calculate prediction volatility
            returns = np.diff(predictions) / predictions[:-1]
            volatility = np.std(returns)
            
            # Get historical volatility for comparison
            historical_data = self.data_storage.get_latest_data(symbol, hours_back=168)  # 1 week
            if not historical_data.empty:
                hist_returns = historical_data['close'].pct_change().dropna()
                hist_volatility = hist_returns.std()
                
                # Compare with historical volatility
                volatility_ratio = volatility / hist_volatility if hist_volatility > 0 else 1.0
            else:
                hist_volatility = None
                volatility_ratio = 1.0
            
            # Calculate score
            if volatility <= self.volatility_threshold:
                status = ValidationStatus.VALID
                score = 1.0
                message = f"Volatility within normal range ({volatility:.3f})"
            elif volatility <= self.volatility_threshold * 2:
                status = ValidationStatus.WARNING
                score = 0.6
                message = f"Elevated volatility detected ({volatility:.3f})"
            else:
                status = ValidationStatus.INVALID
                score = 0.0
                message = f"Excessive volatility detected ({volatility:.3f})"
            
            return ValidationResult(
                ValidationType.VOLATILITY,
                status,
                score,
                message,
                {
                    'prediction_volatility': volatility,
                    'historical_volatility': hist_volatility,
                    'volatility_ratio': volatility_ratio
                }
            )
            
        except Exception as e:
            logger.error(f"Error in volatility validation: {e}")
            return ValidationResult(
                ValidationType.VOLATILITY,
                ValidationStatus.INVALID,
                0.0,
                f"Volatility validation failed: {e}",
                {}
            )
    
    def _validate_trend_consistency(self, symbol: str, prediction_result: Dict[str, Any]) -> ValidationResult:
        """Validate trend consistency"""
        try:
            predictions_df = prediction_result['predictions']
            predictions = predictions_df['predicted_price'].values
            
            # Get recent historical data
            historical_data = self.data_storage.get_latest_data(symbol, hours_back=24)
            if historical_data.empty:
                return ValidationResult(
                    ValidationType.TREND_CONSISTENCY,
                    ValidationStatus.WARNING,
                    0.5,
                    "Insufficient historical data for trend analysis",
                    {}
                )
            
            # Calculate historical trend
            hist_prices = historical_data['close'].values
            hist_trend = np.polyfit(range(len(hist_prices)), hist_prices, 1)[0]
            
            # Calculate prediction trend
            pred_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            
            # Check trend consistency
            if hist_trend == 0:
                trend_consistency = 1.0 if abs(pred_trend) < 0.01 else 0.5
            else:
                trend_consistency = 1.0 - abs((pred_trend - hist_trend) / hist_trend)
            
            # Calculate score
            if trend_consistency >= self.trend_consistency_threshold:
                status = ValidationStatus.VALID
                score = trend_consistency
                message = f"Trend consistency good ({trend_consistency:.2f})"
            elif trend_consistency >= 0.5:
                status = ValidationStatus.WARNING
                score = trend_consistency
                message = f"Moderate trend consistency ({trend_consistency:.2f})"
            else:
                status = ValidationStatus.INVALID
                score = trend_consistency
                message = f"Poor trend consistency ({trend_consistency:.2f})"
            
            return ValidationResult(
                ValidationType.TREND_CONSISTENCY,
                status,
                score,
                message,
                {
                    'historical_trend': hist_trend,
                    'prediction_trend': pred_trend,
                    'trend_consistency': trend_consistency
                }
            )
            
        except Exception as e:
            logger.error(f"Error in trend consistency validation: {e}")
            return ValidationResult(
                ValidationType.TREND_CONSISTENCY,
                ValidationStatus.INVALID,
                0.0,
                f"Trend consistency validation failed: {e}",
                {}
            )
    
    def _validate_confidence_threshold(self, symbol: str, prediction_result: Dict[str, Any]) -> ValidationResult:
        """Validate confidence threshold"""
        try:
            confidence_score = prediction_result.get('confidence_score', 0.0)
            
            # Calculate score based on confidence
            if confidence_score >= 0.8:
                status = ValidationStatus.VALID
                score = confidence_score
                message = f"High confidence prediction ({confidence_score:.3f})"
            elif confidence_score >= self.confidence_threshold:
                status = ValidationStatus.WARNING
                score = confidence_score
                message = f"Moderate confidence prediction ({confidence_score:.3f})"
            else:
                status = ValidationStatus.INVALID
                score = confidence_score
                message = f"Low confidence prediction ({confidence_score:.3f})"
            
            return ValidationResult(
                ValidationType.CONFIDENCE_THRESHOLD,
                status,
                score,
                message,
                {
                    'confidence_score': confidence_score,
                    'threshold': self.confidence_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error in confidence validation: {e}")
            return ValidationResult(
                ValidationType.CONFIDENCE_THRESHOLD,
                ValidationStatus.INVALID,
                0.0,
                f"Confidence validation failed: {e}",
                {}
            )
    
    def _validate_historical_accuracy(self, symbol: str, prediction_result: Dict[str, Any]) -> ValidationResult:
        """Validate based on historical prediction accuracy"""
        try:
            # Get recent prediction accuracy
            recent_accuracy = self._get_recent_prediction_accuracy(symbol, days_back=7)
            
            if recent_accuracy is None:
                return ValidationResult(
                    ValidationType.HISTORICAL_ACCURACY,
                    ValidationStatus.WARNING,
                    0.5,
                    "No historical accuracy data available",
                    {}
                )
            
            # Calculate score
            if recent_accuracy >= 0.8:
                status = ValidationStatus.VALID
                score = recent_accuracy
                message = f"Good historical accuracy ({recent_accuracy:.2f})"
            elif recent_accuracy >= self.historical_accuracy_threshold:
                status = ValidationStatus.WARNING
                score = recent_accuracy
                message = f"Moderate historical accuracy ({recent_accuracy:.2f})"
            else:
                status = ValidationStatus.INVALID
                score = recent_accuracy
                message = f"Poor historical accuracy ({recent_accuracy:.2f})"
            
            return ValidationResult(
                ValidationType.HISTORICAL_ACCURACY,
                status,
                score,
                message,
                {
                    'recent_accuracy': recent_accuracy,
                    'threshold': self.historical_accuracy_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error in historical accuracy validation: {e}")
            return ValidationResult(
                ValidationType.HISTORICAL_ACCURACY,
                ValidationStatus.INVALID,
                0.0,
                f"Historical accuracy validation failed: {e}",
                {}
            )
    
    def _validate_market_conditions(self, symbol: str, prediction_result: Dict[str, Any]) -> ValidationResult:
        """Validate based on current market conditions"""
        try:
            # Get market-wide data (using a market index as proxy)
            market_data = self.data_storage.get_latest_data('^GSPC', hours_back=24)  # S&P 500
            if market_data.empty:
                return ValidationResult(
                    ValidationType.MARKET_CONDITIONS,
                    ValidationStatus.WARNING,
                    0.5,
                    "No market data available for validation",
                    {}
                )
            
            # Calculate market volatility
            market_returns = market_data['close'].pct_change().dropna()
            market_volatility = market_returns.std()
            
            # Check for market anomalies
            volume_data = market_data['volume']
            avg_volume = volume_data.mean()
            current_volume = volume_data.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate score
            score = 1.0
            status = ValidationStatus.VALID
            issues = []
            
            if market_volatility > self.market_volatility_threshold:
                score -= 0.3
                issues.append(f"High market volatility ({market_volatility:.3f})")
            
            if volume_ratio > self.volume_anomaly_threshold:
                score -= 0.2
                issues.append(f"Volume anomaly ({volume_ratio:.1f}x normal)")
            
            if score >= 0.8:
                status = ValidationStatus.VALID
                message = "Normal market conditions"
            elif score >= 0.6:
                status = ValidationStatus.WARNING
                message = f"Elevated market conditions: {', '.join(issues)}"
            else:
                status = ValidationStatus.INVALID
                message = f"Abnormal market conditions: {', '.join(issues)}"
            
            return ValidationResult(
                ValidationType.MARKET_CONDITIONS,
                status,
                score,
                message,
                {
                    'market_volatility': market_volatility,
                    'volume_ratio': volume_ratio,
                    'market_issues': issues
                }
            )
            
        except Exception as e:
            logger.error(f"Error in market conditions validation: {e}")
            return ValidationResult(
                ValidationType.MARKET_CONDITIONS,
                ValidationStatus.WARNING,
                0.5,
                f"Market conditions validation failed: {e}",
                {}
            )
    
    def _get_recent_prediction_accuracy(self, symbol: str, days_back: int = 7) -> Optional[float]:
        """Get recent prediction accuracy for a symbol"""
        try:
            # Get recent predictions and actual prices
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # This would typically query the database for historical predictions
            # For now, return a placeholder
            return 0.75  # Placeholder accuracy
            
        except Exception as e:
            logger.error(f"Error getting recent accuracy for {symbol}: {e}")
            return None
    
    def _calculate_overall_validation(self, symbol: str, prediction_result: Dict[str, Any], 
                                    validation_results: List[ValidationResult]) -> OverallValidationResult:
        """Calculate overall validation result"""
        try:
            # Calculate weighted score
            total_score = 0.0
            total_weight = 0.0
            
            for result in validation_results:
                weight = self.validation_weights.get(result.validation_type, 0.1)
                total_score += result.score * weight
                total_weight += weight
            
            overall_score = total_score / total_weight if total_weight > 0 else 0.0
            
            # Determine overall status
            critical_count = sum(1 for r in validation_results if r.status == ValidationStatus.CRITICAL)
            invalid_count = sum(1 for r in validation_results if r.status == ValidationStatus.INVALID)
            warning_count = sum(1 for r in validation_results if r.status == ValidationStatus.WARNING)
            
            if critical_count > 0:
                overall_status = ValidationStatus.CRITICAL
            elif invalid_count > 2:
                overall_status = ValidationStatus.INVALID
            elif invalid_count > 0 or warning_count > 3:
                overall_status = ValidationStatus.WARNING
            else:
                overall_status = ValidationStatus.VALID
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results, overall_score)
            
            # Determine if reprediction is required
            requires_reprediction = (
                overall_status in [ValidationStatus.CRITICAL, ValidationStatus.INVALID] or
                overall_score < 0.5
            )
            
            # Calculate confidence adjustment
            confidence_adjustment = self._calculate_confidence_adjustment(validation_results)
            
            return OverallValidationResult(
                symbol=symbol,
                prediction_timestamp=prediction_result.get('prediction_timestamp', datetime.now()),
                overall_status=overall_status,
                overall_score=overall_score,
                validation_results=validation_results,
                recommendations=recommendations,
                requires_reprediction=requires_reprediction,
                confidence_adjustment=confidence_adjustment
            )
            
        except Exception as e:
            logger.error(f"Error calculating overall validation: {e}")
            return self._create_error_validation_result(symbol, str(e))
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                overall_score: float) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for result in validation_results:
            if result.status == ValidationStatus.CRITICAL:
                recommendations.append(f"CRITICAL: {result.message}")
            elif result.status == ValidationStatus.INVALID:
                recommendations.append(f"INVALID: {result.message}")
            elif result.status == ValidationStatus.WARNING:
                recommendations.append(f"WARNING: {result.message}")
        
        if overall_score < 0.5:
            recommendations.append("Consider retraining the model")
        
        if not recommendations:
            recommendations.append("All validations passed successfully")
        
        return recommendations
    
    def _calculate_confidence_adjustment(self, validation_results: List[ValidationResult]) -> float:
        """Calculate confidence adjustment based on validation results"""
        adjustment = 0.0
        
        for result in validation_results:
            if result.status == ValidationStatus.CRITICAL:
                adjustment -= 0.3
            elif result.status == ValidationStatus.INVALID:
                adjustment -= 0.2
            elif result.status == ValidationStatus.WARNING:
                adjustment -= 0.1
        
        return max(-0.5, min(0.2, adjustment))  # Limit adjustment between -50% and +20%
    
    def _store_validation_results(self, overall_result: OverallValidationResult):
        """Store validation results in database"""
        try:
            # Store overall validation result
            self.data_storage.store_validation_result(
                symbol=overall_result.symbol,
                prediction_timestamp=overall_result.prediction_timestamp,
                overall_status=overall_result.overall_status.value,
                overall_score=overall_result.overall_score,
                requires_reprediction=overall_result.requires_reprediction,
                confidence_adjustment=overall_result.confidence_adjustment,
                recommendations=overall_result.recommendations
            )
            
            # Store individual validation results
            for result in overall_result.validation_results:
                self.data_storage.store_validation_detail(
                    symbol=overall_result.symbol,
                    prediction_timestamp=overall_result.prediction_timestamp,
                    validation_type=result.validation_type.value,
                    status=result.status.value,
                    score=result.score,
                    message=result.message,
                    details=result.details
                )
            
        except Exception as e:
            logger.error(f"Error storing validation results: {e}")
    
    def _create_error_validation_result(self, symbol: str, error_message: str) -> OverallValidationResult:
        """Create error validation result"""
        return OverallValidationResult(
            symbol=symbol,
            prediction_timestamp=datetime.now(),
            overall_status=ValidationStatus.CRITICAL,
            overall_score=0.0,
            validation_results=[],
            recommendations=[f"Validation failed: {error_message}"],
            requires_reprediction=True,
            confidence_adjustment=-0.5
        )


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("PredictionValidationWorkflow module loaded successfully")
    print("Use with DataStorage and ModelManager instances for full functionality")
