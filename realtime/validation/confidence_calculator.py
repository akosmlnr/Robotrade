"""
Advanced Confidence Calculator for Real-time LSTM Prediction System
Phase 2.2: Multi-factor confidence calculation with uncertainty quantification
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    VERY_LOW = "very_low"      # 0.0 - 0.3
    LOW = "low"                # 0.3 - 0.5
    MEDIUM = "medium"          # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.85
    VERY_HIGH = "very_high"    # 0.85 - 1.0

@dataclass
class ConfidenceFactors:
    """Individual confidence factors"""
    model_performance: float = 0.0
    data_quality: float = 0.0
    market_volatility: float = 0.0
    trend_consistency: float = 0.0
    volume_analysis: float = 0.0
    technical_indicators: float = 0.0
    time_of_day: float = 0.0
    market_regime: float = 0.0
    prediction_stability: float = 0.0
    historical_accuracy: float = 0.0

@dataclass
class ConfidenceResult:
    """Comprehensive confidence calculation result"""
    overall_confidence: float
    confidence_level: ConfidenceLevel
    factors: ConfidenceFactors
    uncertainty_estimate: float
    recommendation: str
    timestamp: datetime
    symbol: str

class AdvancedConfidenceCalculator:
    """
    Advanced confidence calculator with multiple factors and uncertainty quantification
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the confidence calculator
        
        Args:
            config: Configuration dictionary with confidence parameters
        """
        self.config = config
        self.factor_weights = config.get('confidence_factor_weights', {
            'model_performance': 0.25,
            'data_quality': 0.15,
            'market_volatility': 0.15,
            'trend_consistency': 0.10,
            'volume_analysis': 0.10,
            'technical_indicators': 0.10,
            'time_of_day': 0.05,
            'market_regime': 0.05,
            'prediction_stability': 0.03,
            'historical_accuracy': 0.02
        })
        
        # Confidence thresholds
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_LOW: (0.0, 0.3),
            ConfidenceLevel.LOW: (0.3, 0.5),
            ConfidenceLevel.MEDIUM: (0.5, 0.7),
            ConfidenceLevel.HIGH: (0.7, 0.85),
            ConfidenceLevel.VERY_HIGH: (0.85, 1.0)
        }
        
        # Historical performance tracking
        self.performance_history = {}
        
        logger.info("AdvancedConfidenceCalculator initialized")
    
    def calculate_confidence(self, symbol: str, predictions: List[float], 
                           market_data: pd.DataFrame, model_performance: Dict[str, float],
                           prediction_timestamp: datetime) -> ConfidenceResult:
        """
        Calculate comprehensive confidence score for predictions
        
        Args:
            symbol: Stock symbol
            predictions: List of predicted prices
            market_data: Recent market data
            model_performance: Model performance metrics
            prediction_timestamp: When prediction was made
            
        Returns:
            ConfidenceResult with detailed confidence analysis
        """
        try:
            # Calculate individual confidence factors
            factors = self._calculate_all_factors(
                symbol, predictions, market_data, model_performance, prediction_timestamp
            )
            
            # Calculate weighted overall confidence
            overall_confidence = self._calculate_weighted_confidence(factors)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_confidence)
            
            # Calculate uncertainty estimate
            uncertainty_estimate = self._calculate_uncertainty(factors, overall_confidence)
            
            # Generate recommendation
            recommendation = self._generate_confidence_recommendation(
                confidence_level, uncertainty_estimate, factors
            )
            
            result = ConfidenceResult(
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                factors=factors,
                uncertainty_estimate=uncertainty_estimate,
                recommendation=recommendation,
                timestamp=prediction_timestamp,
                symbol=symbol
            )
            
            # Store performance history
            self._store_confidence_result(symbol, result)
            
            logger.info(f"Confidence calculated for {symbol}: {confidence_level.value} ({overall_confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating confidence for {symbol}: {e}")
            return self._create_default_confidence_result(symbol, prediction_timestamp)
    
    def _calculate_all_factors(self, symbol: str, predictions: List[float], 
                             market_data: pd.DataFrame, model_performance: Dict[str, float],
                             prediction_timestamp: datetime) -> ConfidenceFactors:
        """Calculate all individual confidence factors"""
        try:
            factors = ConfidenceFactors()
            
            # Model performance factor
            factors.model_performance = self._calculate_model_performance_factor(model_performance)
            
            # Data quality factor
            factors.data_quality = self._calculate_data_quality_factor(market_data)
            
            # Market volatility factor
            factors.market_volatility = self._calculate_volatility_factor(market_data)
            
            # Trend consistency factor
            factors.trend_consistency = self._calculate_trend_consistency_factor(market_data, predictions)
            
            # Volume analysis factor
            factors.volume_analysis = self._calculate_volume_analysis_factor(market_data)
            
            # Technical indicators factor
            factors.technical_indicators = self._calculate_technical_indicators_factor(market_data)
            
            # Time of day factor
            factors.time_of_day = self._calculate_time_of_day_factor(prediction_timestamp)
            
            # Market regime factor
            factors.market_regime = self._calculate_market_regime_factor(market_data)
            
            # Prediction stability factor
            factors.prediction_stability = self._calculate_prediction_stability_factor(predictions)
            
            # Historical accuracy factor
            factors.historical_accuracy = self._calculate_historical_accuracy_factor(symbol)
            
            return factors
            
        except Exception as e:
            logger.error(f"Error calculating confidence factors for {symbol}: {e}")
            return ConfidenceFactors()
    
    def _calculate_model_performance_factor(self, model_performance: Dict[str, float]) -> float:
        """Calculate confidence based on model performance metrics"""
        try:
            # Get key performance metrics
            rmse = model_performance.get('rmse', 0.1)
            r2 = model_performance.get('r2', 0.0)
            accuracy = model_performance.get('accuracy', 0.5)
            
            # Normalize RMSE (lower is better, assume 0.05 is excellent, 0.2 is poor)
            rmse_factor = max(0.0, 1.0 - (rmse / 0.2))
            
            # R² factor (higher is better)
            r2_factor = max(0.0, r2)
            
            # Accuracy factor
            accuracy_factor = accuracy
            
            # Combine factors
            performance_score = (rmse_factor * 0.4 + r2_factor * 0.4 + accuracy_factor * 0.2)
            
            return max(0.0, min(1.0, performance_score))
            
        except Exception as e:
            logger.error(f"Error calculating model performance factor: {e}")
            return 0.5
    
    def _calculate_data_quality_factor(self, market_data: pd.DataFrame) -> float:
        """Calculate confidence based on data quality"""
        try:
            if market_data.empty:
                return 0.0
            
            quality_score = 1.0
            
            # Check for missing values
            missing_ratio = market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))
            quality_score -= missing_ratio * 0.5
            
            # Check for outliers (using IQR method)
            for column in ['close', 'volume']:
                if column in market_data.columns:
                    Q1 = market_data[column].quantile(0.25)
                    Q3 = market_data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((market_data[column] < (Q1 - 1.5 * IQR)) | 
                              (market_data[column] > (Q3 + 1.5 * IQR))).sum()
                    outlier_ratio = outliers / len(market_data)
                    quality_score -= outlier_ratio * 0.3
            
            # Check for data consistency
            if 'close' in market_data.columns:
                # Check for negative prices
                negative_prices = (market_data['close'] <= 0).sum()
                if negative_prices > 0:
                    quality_score -= 0.5
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating data quality factor: {e}")
            return 0.5
    
    def _calculate_volatility_factor(self, market_data: pd.DataFrame) -> float:
        """Calculate confidence based on market volatility"""
        try:
            if len(market_data) < 2:
                return 0.5
            
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.std()
            
            # Normalize volatility (assume 0.02 is normal, 0.1 is very high)
            volatility_factor = max(0.0, 1.0 - (volatility / 0.1))
            
            return max(0.0, min(1.0, volatility_factor))
            
        except Exception as e:
            logger.error(f"Error calculating volatility factor: {e}")
            return 0.5
    
    def _calculate_trend_consistency_factor(self, market_data: pd.DataFrame, predictions: List[float]) -> float:
        """Calculate confidence based on trend consistency"""
        try:
            if len(market_data) < 5 or len(predictions) < 3:
                return 0.5
            
            # Calculate recent trend in market data
            recent_prices = market_data['close'].tail(5)
            market_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Calculate trend in predictions
            pred_trend = (predictions[-1] - predictions[0]) / predictions[0]
            
            # Check trend consistency
            trend_consistency = 1.0 - abs(market_trend - pred_trend)
            
            return max(0.0, min(1.0, trend_consistency))
            
        except Exception as e:
            logger.error(f"Error calculating trend consistency factor: {e}")
            return 0.5
    
    def _calculate_volume_analysis_factor(self, market_data: pd.DataFrame) -> float:
        """Calculate confidence based on volume analysis"""
        try:
            if 'volume' not in market_data.columns or len(market_data) < 5:
                return 0.5
            
            # Calculate volume trend
            recent_volume = market_data['volume'].tail(5)
            avg_volume = recent_volume.mean()
            volume_trend = (recent_volume.iloc[-1] - recent_volume.iloc[0]) / recent_volume.iloc[0]
            
            # Normal volume range (not too low, not too high)
            volume_factor = 1.0
            
            # Check for unusually low volume
            if avg_volume < recent_volume.quantile(0.1):
                volume_factor -= 0.3
            
            # Check for unusually high volume
            if avg_volume > recent_volume.quantile(0.9):
                volume_factor -= 0.2
            
            # Check for extreme volume changes
            if abs(volume_trend) > 0.5:  # 50% change
                volume_factor -= 0.2
            
            return max(0.0, min(1.0, volume_factor))
            
        except Exception as e:
            logger.error(f"Error calculating volume analysis factor: {e}")
            return 0.5
    
    def _calculate_technical_indicators_factor(self, market_data: pd.DataFrame) -> float:
        """Calculate confidence based on technical indicators alignment"""
        try:
            if len(market_data) < 20:
                return 0.5
            
            # Simple technical indicators
            close_prices = market_data['close']
            
            # Moving averages
            sma_10 = close_prices.rolling(10).mean()
            sma_20 = close_prices.rolling(20).mean()
            
            # RSI (simplified)
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate alignment score
            alignment_score = 0.5
            
            # Price vs moving averages
            current_price = close_prices.iloc[-1]
            if not pd.isna(sma_10.iloc[-1]) and not pd.isna(sma_20.iloc[-1]):
                if current_price > sma_10.iloc[-1] > sma_20.iloc[-1]:
                    alignment_score += 0.2  # Bullish alignment
                elif current_price < sma_10.iloc[-1] < sma_20.iloc[-1]:
                    alignment_score += 0.2  # Bearish alignment
                else:
                    alignment_score -= 0.1  # Mixed signals
            
            # RSI analysis
            if not pd.isna(rsi.iloc[-1]):
                rsi_value = rsi.iloc[-1]
                if 30 < rsi_value < 70:  # Normal range
                    alignment_score += 0.1
                elif rsi_value < 20 or rsi_value > 80:  # Extreme values
                    alignment_score -= 0.1
            
            return max(0.0, min(1.0, alignment_score))
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators factor: {e}")
            return 0.5
    
    def _calculate_time_of_day_factor(self, prediction_timestamp: datetime) -> float:
        """Calculate confidence based on time of day"""
        try:
            hour = prediction_timestamp.hour
            
            # Market hours confidence (assuming US market 9:30 AM - 4:00 PM EST)
            if 9 <= hour <= 16:
                return 1.0  # Market hours
            elif 8 <= hour <= 17:
                return 0.8  # Extended hours
            elif 6 <= hour <= 20:
                return 0.6  # Pre/post market
            else:
                return 0.4  # Off hours
            
        except Exception as e:
            logger.error(f"Error calculating time of day factor: {e}")
            return 0.5
    
    def _calculate_market_regime_factor(self, market_data: pd.DataFrame) -> float:
        """Calculate confidence based on market regime"""
        try:
            if len(market_data) < 20:
                return 0.5
            
            # Calculate market regime indicators
            close_prices = market_data['close']
            
            # Volatility regime
            returns = close_prices.pct_change().dropna()
            volatility = returns.rolling(20).std()
            current_vol = volatility.iloc[-1]
            
            # Trend regime
            sma_20 = close_prices.rolling(20).mean()
            trend_strength = abs(close_prices.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
            
            # Regime score
            regime_score = 0.5
            
            # Low volatility = higher confidence
            if current_vol < 0.02:
                regime_score += 0.2
            elif current_vol > 0.05:
                regime_score -= 0.2
            
            # Moderate trend = higher confidence
            if 0.01 < trend_strength < 0.05:
                regime_score += 0.1
            elif trend_strength > 0.1:
                regime_score -= 0.1
            
            return max(0.0, min(1.0, regime_score))
            
        except Exception as e:
            logger.error(f"Error calculating market regime factor: {e}")
            return 0.5
    
    def _calculate_prediction_stability_factor(self, predictions: List[float]) -> float:
        """Calculate confidence based on prediction stability"""
        try:
            if len(predictions) < 3:
                return 0.5
            
            # Calculate prediction variance
            pred_array = np.array(predictions)
            prediction_variance = np.var(pred_array)
            mean_prediction = np.mean(pred_array)
            
            # Normalize variance
            if mean_prediction > 0:
                normalized_variance = prediction_variance / (mean_prediction ** 2)
                stability_score = max(0.0, 1.0 - (normalized_variance * 10))
            else:
                stability_score = 0.5
            
            return max(0.0, min(1.0, stability_score))
            
        except Exception as e:
            logger.error(f"Error calculating prediction stability factor: {e}")
            return 0.5
    
    def _calculate_historical_accuracy_factor(self, symbol: str) -> float:
        """Calculate confidence based on historical accuracy"""
        try:
            if symbol not in self.performance_history:
                return 0.5
            
            recent_results = self.performance_history[symbol][-20:]  # Last 20 predictions
            
            if not recent_results:
                return 0.5
            
            # Calculate average confidence from recent results
            avg_confidence = np.mean([r.overall_confidence for r in recent_results])
            
            return float(avg_confidence) if not np.isnan(avg_confidence) else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating historical accuracy factor for {symbol}: {e}")
            return 0.5
    
    def _calculate_weighted_confidence(self, factors: ConfidenceFactors) -> float:
        """Calculate weighted overall confidence from individual factors"""
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for factor_name, weight in self.factor_weights.items():
                factor_value = getattr(factors, factor_name, 0.5)
                weighted_sum += factor_value * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating weighted confidence: {e}")
            return 0.5
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from score"""
        try:
            for level, (min_val, max_val) in self.confidence_thresholds.items():
                if min_val <= confidence_score < max_val:
                    return level
            
            # Handle edge case
            if confidence_score >= 1.0:
                return ConfidenceLevel.VERY_HIGH
            else:
                return ConfidenceLevel.VERY_LOW
                
        except Exception as e:
            logger.error(f"Error determining confidence level: {e}")
            return ConfidenceLevel.MEDIUM
    
    def _calculate_uncertainty(self, factors: ConfidenceFactors, overall_confidence: float) -> float:
        """Calculate uncertainty estimate"""
        try:
            # Calculate factor variance as uncertainty measure
            factor_values = [
                factors.model_performance,
                factors.data_quality,
                factors.market_volatility,
                factors.trend_consistency,
                factors.volume_analysis,
                factors.technical_indicators,
                factors.time_of_day,
                factors.market_regime,
                factors.prediction_stability,
                factors.historical_accuracy
            ]
            
            factor_variance = np.var(factor_values)
            uncertainty = min(1.0, factor_variance * 2)  # Scale variance to uncertainty
            
            return float(uncertainty) if not np.isnan(uncertainty) else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty: {e}")
            return 0.5
    
    def _generate_confidence_recommendation(self, confidence_level: ConfidenceLevel, 
                                         uncertainty: float, factors: ConfidenceFactors) -> str:
        """Generate recommendation based on confidence analysis"""
        try:
            recommendations = {
                ConfidenceLevel.VERY_HIGH: "Very high confidence - Strong trading signal",
                ConfidenceLevel.HIGH: "High confidence - Good trading signal",
                ConfidenceLevel.MEDIUM: "Medium confidence - Moderate trading signal",
                ConfidenceLevel.LOW: "Low confidence - Weak trading signal",
                ConfidenceLevel.VERY_LOW: "Very low confidence - Avoid trading"
            }
            
            base_recommendation = recommendations.get(confidence_level, "Unknown confidence level")
            
            # Add uncertainty warning
            if uncertainty > 0.3:
                base_recommendation += " (High uncertainty detected)"
            
            # Add specific factor warnings
            warnings = []
            if factors.data_quality < 0.5:
                warnings.append("Poor data quality")
            if factors.market_volatility < 0.3:
                warnings.append("High market volatility")
            if factors.model_performance < 0.5:
                warnings.append("Poor model performance")
            
            if warnings:
                base_recommendation += f" - Warnings: {', '.join(warnings)}"
            
            return base_recommendation
            
        except Exception as e:
            logger.error(f"Error generating confidence recommendation: {e}")
            return "Confidence analysis failed"
    
    def _store_confidence_result(self, symbol: str, result: ConfidenceResult):
        """Store confidence result in history"""
        try:
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []
            
            self.performance_history[symbol].append(result)
            
            # Keep only recent results (last 100)
            if len(self.performance_history[symbol]) > 100:
                self.performance_history[symbol] = self.performance_history[symbol][-100:]
                
        except Exception as e:
            logger.error(f"Error storing confidence result for {symbol}: {e}")
    
    def _create_default_confidence_result(self, symbol: str, timestamp: datetime) -> ConfidenceResult:
        """Create default confidence result for error cases"""
        return ConfidenceResult(
            overall_confidence=0.5,
            confidence_level=ConfidenceLevel.MEDIUM,
            factors=ConfidenceFactors(),
            uncertainty_estimate=0.5,
            recommendation="Confidence calculation failed - using default values",
            timestamp=timestamp,
            symbol=symbol
        )
    
    def get_confidence_summary(self, symbol: str) -> Dict[str, Any]:
        """Get confidence summary for a symbol"""
        try:
            if symbol not in self.performance_history:
                return {'symbol': symbol, 'total_calculations': 0}
            
            results = self.performance_history[symbol]
            
            return {
                'symbol': symbol,
                'total_calculations': len(results),
                'avg_confidence': np.mean([r.overall_confidence for r in results]),
                'avg_uncertainty': np.mean([r.uncertainty_estimate for r in results]),
                'confidence_level_distribution': {
                    level.value: sum(1 for r in results if r.confidence_level == level)
                    for level in ConfidenceLevel
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting confidence summary for {symbol}: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = {
        'confidence_factor_weights': {
            'model_performance': 0.25,
            'data_quality': 0.15,
            'market_volatility': 0.15,
            'trend_consistency': 0.10,
            'volume_analysis': 0.10,
            'technical_indicators': 0.10,
            'time_of_day': 0.05,
            'market_regime': 0.05,
            'prediction_stability': 0.03,
            'historical_accuracy': 0.02
        }
    }
    
    # Test the confidence calculator
    calculator = AdvancedConfidenceCalculator(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'close': [100, 101, 102, 101, 103, 102, 104, 105, 103, 106],
        'volume': [1000, 1100, 1200, 1000, 1300, 1100, 1400, 1500, 1200, 1600]
    })
    
    sample_predictions = [105, 106, 107, 108, 109]
    model_performance = {'rmse': 0.02, 'r2': 0.8, 'accuracy': 0.75}
    
    # Test confidence calculation
    result = calculator.calculate_confidence(
        symbol='AAPL',
        predictions=sample_predictions,
        market_data=sample_data,
        model_performance=model_performance,
        prediction_timestamp=datetime.now()
    )
    
    print(f"Confidence Level: {result.confidence_level.value}")
    print(f"Overall Confidence: {result.overall_confidence:.3f}")
    print(f"Uncertainty: {result.uncertainty_estimate:.3f}")
    print(f"Recommendation: {result.recommendation}")
    
    print("Advanced confidence calculator test completed")
