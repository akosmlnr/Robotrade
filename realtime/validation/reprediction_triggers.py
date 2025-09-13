"""
Reprediction Triggers for Real-time LSTM Prediction System
Phase 3.1: Intelligent triggers for when to regenerate predictions
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class TriggerType(Enum):
    """Types of reprediction triggers"""
    VALIDATION_FAILURE = "validation_failure"
    MARKET_VOLATILITY = "market_volatility"
    PRICE_ANOMALY = "price_anomaly"
    VOLUME_SPIKE = "volume_spike"
    TIME_BASED = "time_based"
    MODEL_DRIFT = "model_drift"
    EXTERNAL_EVENT = "external_event"
    MANUAL = "manual"

class TriggerPriority(Enum):
    """Priority levels for triggers"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TriggerEvent:
    """Represents a trigger event"""
    trigger_id: str
    trigger_type: TriggerType
    symbol: str
    priority: TriggerPriority
    timestamp: datetime
    reason: str
    details: Dict[str, Any]
    processed: bool = False
    processed_at: Optional[datetime] = None

class RepredictionTriggers:
    """
    Intelligent system for triggering repredictions based on various conditions
    """
    
    def __init__(self, data_storage, validation_workflow, update_scheduler):
        """
        Initialize reprediction triggers
        
        Args:
            data_storage: DataStorage instance
            validation_workflow: ValidationWorkflow instance
            update_scheduler: UpdateScheduler instance
        """
        self.data_storage = data_storage
        self.validation_workflow = validation_workflow
        self.update_scheduler = update_scheduler
        
        # Trigger thresholds
        self.volatility_threshold = 0.08  # 8% volatility threshold
        self.price_change_threshold = 0.10  # 10% price change threshold
        self.volume_spike_threshold = 2.5   # 2.5x normal volume
        self.model_drift_threshold = 0.15   # 15% model performance degradation
        
        # Time-based triggers
        self.emergency_update_interval = 5   # 5 minutes for critical events
        self.urgent_update_interval = 15     # 15 minutes for urgent events
        self.normal_update_interval = 60     # 60 minutes for normal events
        
        # Active triggers
        self.active_triggers: Dict[str, TriggerEvent] = {}
        self.trigger_history: List[TriggerEvent] = []
        
        # Callbacks
        self.on_trigger_activated: Optional[Callable] = None
        self.on_reprediction_requested: Optional[Callable] = None
        
        # Trigger statistics
        self.trigger_stats = {
            'total_triggers': 0,
            'processed_triggers': 0,
            'failed_triggers': 0,
            'triggers_by_type': {},
            'triggers_by_priority': {}
        }
        
        logger.info("RepredictionTriggers initialized")
    
    def check_validation_failure_trigger(self, symbol: str, validation_result: Any) -> Optional[TriggerEvent]:
        """
        Check if validation failure should trigger reprediction
        
        Args:
            symbol: Stock symbol
            validation_result: Validation result from ValidationWorkflow
            
        Returns:
            TriggerEvent if triggered, None otherwise
        """
        try:
            if validation_result.requires_reprediction:
                trigger_id = f"validation_failure_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Determine priority based on validation status
                if validation_result.overall_status.value == "critical":
                    priority = TriggerPriority.CRITICAL
                elif validation_result.overall_status.value == "invalid":
                    priority = TriggerPriority.HIGH
                else:
                    priority = TriggerPriority.MEDIUM
                
                trigger = TriggerEvent(
                    trigger_id=trigger_id,
                    trigger_type=TriggerType.VALIDATION_FAILURE,
                    symbol=symbol,
                    priority=priority,
                    timestamp=datetime.now(),
                    reason=f"Validation failed: {validation_result.overall_status.value}",
                    details={
                        'overall_score': validation_result.overall_score,
                        'validation_status': validation_result.overall_status.value,
                        'recommendations': validation_result.recommendations
                    }
                )
                
                self._activate_trigger(trigger)
                return trigger
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking validation failure trigger for {symbol}: {e}")
            return None
    
    def check_market_volatility_trigger(self, symbol: str) -> Optional[TriggerEvent]:
        """
        Check if market volatility should trigger reprediction
        
        Args:
            symbol: Stock symbol
            
        Returns:
            TriggerEvent if triggered, None otherwise
        """
        try:
            # Get recent market data
            market_data = self.data_storage.get_latest_data(symbol, hours_back=24)
            if market_data.empty:
                return None
            
            # Calculate recent volatility
            returns = market_data['close'].pct_change().dropna()
            recent_volatility = returns.tail(20).std()  # Last 20 periods
            
            if recent_volatility > self.volatility_threshold:
                trigger_id = f"volatility_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Determine priority based on volatility level
                if recent_volatility > self.volatility_threshold * 2:
                    priority = TriggerPriority.CRITICAL
                elif recent_volatility > self.volatility_threshold * 1.5:
                    priority = TriggerPriority.HIGH
                else:
                    priority = TriggerPriority.MEDIUM
                
                trigger = TriggerEvent(
                    trigger_id=trigger_id,
                    trigger_type=TriggerType.MARKET_VOLATILITY,
                    symbol=symbol,
                    priority=priority,
                    timestamp=datetime.now(),
                    reason=f"High market volatility detected: {recent_volatility:.3f}",
                    details={
                        'volatility': recent_volatility,
                        'threshold': self.volatility_threshold,
                        'volatility_ratio': recent_volatility / self.volatility_threshold
                    }
                )
                
                self._activate_trigger(trigger)
                return trigger
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking market volatility trigger for {symbol}: {e}")
            return None
    
    def check_price_anomaly_trigger(self, symbol: str) -> Optional[TriggerEvent]:
        """
        Check if price anomaly should trigger reprediction
        
        Args:
            symbol: Stock symbol
            
        Returns:
            TriggerEvent if triggered, None otherwise
        """
        try:
            # Get recent price data
            price_data = self.data_storage.get_latest_data(symbol, hours_back=48)
            if price_data.empty:
                return None
            
            current_price = price_data['close'].iloc[-1]
            
            # Calculate price change from various timeframes
            price_changes = {}
            for hours in [1, 4, 12, 24]:
                if len(price_data) > hours:
                    past_price = price_data['close'].iloc[-hours-1]
                    change = abs(current_price - past_price) / past_price
                    price_changes[f'{hours}h'] = change
            
            # Check for significant price changes
            max_change = max(price_changes.values()) if price_changes else 0
            
            if max_change > self.price_change_threshold:
                trigger_id = f"price_anomaly_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Determine priority based on price change magnitude
                if max_change > self.price_change_threshold * 2:
                    priority = TriggerPriority.CRITICAL
                elif max_change > self.price_change_threshold * 1.5:
                    priority = TriggerPriority.HIGH
                else:
                    priority = TriggerPriority.MEDIUM
                
                trigger = TriggerEvent(
                    trigger_id=trigger_id,
                    trigger_type=TriggerType.PRICE_ANOMALY,
                    symbol=symbol,
                    priority=priority,
                    timestamp=datetime.now(),
                    reason=f"Significant price change detected: {max_change:.2%}",
                    details={
                        'current_price': current_price,
                        'price_changes': price_changes,
                        'max_change': max_change,
                        'threshold': self.price_change_threshold
                    }
                )
                
                self._activate_trigger(trigger)
                return trigger
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking price anomaly trigger for {symbol}: {e}")
            return None
    
    def check_volume_spike_trigger(self, symbol: str) -> Optional[TriggerEvent]:
        """
        Check if volume spike should trigger reprediction
        
        Args:
            symbol: Stock symbol
            
        Returns:
            TriggerEvent if triggered, None otherwise
        """
        try:
            # Get recent volume data
            volume_data = self.data_storage.get_latest_data(symbol, hours_back=168)  # 1 week
            if volume_data.empty:
                return None
            
            current_volume = volume_data['volume'].iloc[-1]
            avg_volume = volume_data['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > self.volume_spike_threshold:
                trigger_id = f"volume_spike_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Determine priority based on volume spike magnitude
                if volume_ratio > self.volume_spike_threshold * 2:
                    priority = TriggerPriority.CRITICAL
                elif volume_ratio > self.volume_spike_threshold * 1.5:
                    priority = TriggerPriority.HIGH
                else:
                    priority = TriggerPriority.MEDIUM
                
                trigger = TriggerEvent(
                    trigger_id=trigger_id,
                    trigger_type=TriggerType.VOLUME_SPIKE,
                    symbol=symbol,
                    priority=priority,
                    timestamp=datetime.now(),
                    reason=f"Volume spike detected: {volume_ratio:.1f}x normal",
                    details={
                        'current_volume': current_volume,
                        'average_volume': avg_volume,
                        'volume_ratio': volume_ratio,
                        'threshold': self.volume_spike_threshold
                    }
                )
                
                self._activate_trigger(trigger)
                return trigger
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking volume spike trigger for {symbol}: {e}")
            return None
    
    def check_model_drift_trigger(self, symbol: str) -> Optional[TriggerEvent]:
        """
        Check if model drift should trigger reprediction
        
        Args:
            symbol: Stock symbol
            
        Returns:
            TriggerEvent if triggered, None otherwise
        """
        try:
            # Get recent model performance
            recent_accuracy = self._get_recent_model_accuracy(symbol, days_back=7)
            if recent_accuracy is None:
                return None
            
            # Get baseline accuracy (from model training)
            baseline_accuracy = self._get_baseline_model_accuracy(symbol)
            if baseline_accuracy is None:
                return None
            
            # Calculate performance degradation
            performance_degradation = (baseline_accuracy - recent_accuracy) / baseline_accuracy
            
            if performance_degradation > self.model_drift_threshold:
                trigger_id = f"model_drift_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Determine priority based on degradation level
                if performance_degradation > self.model_drift_threshold * 2:
                    priority = TriggerPriority.CRITICAL
                elif performance_degradation > self.model_drift_threshold * 1.5:
                    priority = TriggerPriority.HIGH
                else:
                    priority = TriggerPriority.MEDIUM
                
                trigger = TriggerEvent(
                    trigger_id=trigger_id,
                    trigger_type=TriggerType.MODEL_DRIFT,
                    symbol=symbol,
                    priority=priority,
                    timestamp=datetime.now(),
                    reason=f"Model drift detected: {performance_degradation:.2%} degradation",
                    details={
                        'recent_accuracy': recent_accuracy,
                        'baseline_accuracy': baseline_accuracy,
                        'performance_degradation': performance_degradation,
                        'threshold': self.model_drift_threshold
                    }
                )
                
                self._activate_trigger(trigger)
                return trigger
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking model drift trigger for {symbol}: {e}")
            return None
    
    def check_time_based_trigger(self, symbol: str, last_prediction_time: datetime) -> Optional[TriggerEvent]:
        """
        Check if time-based conditions should trigger reprediction
        
        Args:
            symbol: Stock symbol
            last_prediction_time: Timestamp of last prediction
            
        Returns:
            TriggerEvent if triggered, None otherwise
        """
        try:
            time_since_last = datetime.now() - last_prediction_time
            
            # Check if it's been too long since last prediction
            if time_since_last > timedelta(hours=2):  # 2 hours without prediction
                trigger_id = f"time_based_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                priority = TriggerPriority.MEDIUM
                if time_since_last > timedelta(hours=6):
                    priority = TriggerPriority.HIGH
                elif time_since_last > timedelta(hours=12):
                    priority = TriggerPriority.CRITICAL
                
                trigger = TriggerEvent(
                    trigger_id=trigger_id,
                    trigger_type=TriggerType.TIME_BASED,
                    symbol=symbol,
                    priority=priority,
                    timestamp=datetime.now(),
                    reason=f"No prediction for {time_since_last}",
                    details={
                        'time_since_last': time_since_last.total_seconds(),
                        'last_prediction_time': last_prediction_time.isoformat()
                    }
                )
                
                self._activate_trigger(trigger)
                return trigger
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking time-based trigger for {symbol}: {e}")
            return None
    
    def create_manual_trigger(self, symbol: str, reason: str, priority: TriggerPriority = TriggerPriority.MEDIUM) -> TriggerEvent:
        """
        Create a manual trigger for reprediction
        
        Args:
            symbol: Stock symbol
            reason: Reason for manual trigger
            priority: Trigger priority
            
        Returns:
            Created TriggerEvent
        """
        try:
            trigger_id = f"manual_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            trigger = TriggerEvent(
                trigger_id=trigger_id,
                trigger_type=TriggerType.MANUAL,
                symbol=symbol,
                priority=priority,
                timestamp=datetime.now(),
                reason=f"Manual trigger: {reason}",
                details={'manual_reason': reason}
            )
            
            self._activate_trigger(trigger)
            return trigger
            
        except Exception as e:
            logger.error(f"Error creating manual trigger for {symbol}: {e}")
            return None
    
    def create_external_event_trigger(self, symbol: str, event_type: str, event_details: Dict[str, Any]) -> TriggerEvent:
        """
        Create a trigger based on external events (news, earnings, etc.)
        
        Args:
            symbol: Stock symbol
            event_type: Type of external event
            event_details: Details about the event
            
        Returns:
            Created TriggerEvent
        """
        try:
            trigger_id = f"external_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine priority based on event type
            priority_map = {
                'earnings': TriggerPriority.HIGH,
                'merger': TriggerPriority.HIGH,
                'news': TriggerPriority.MEDIUM,
                'analyst_upgrade': TriggerPriority.MEDIUM,
                'regulatory': TriggerPriority.HIGH
            }
            priority = priority_map.get(event_type, TriggerPriority.MEDIUM)
            
            trigger = TriggerEvent(
                trigger_id=trigger_id,
                trigger_type=TriggerType.EXTERNAL_EVENT,
                symbol=symbol,
                priority=priority,
                timestamp=datetime.now(),
                reason=f"External event: {event_type}",
                details={
                    'event_type': event_type,
                    'event_details': event_details
                }
            )
            
            self._activate_trigger(trigger)
            return trigger
            
        except Exception as e:
            logger.error(f"Error creating external event trigger for {symbol}: {e}")
            return None
    
    def _activate_trigger(self, trigger: TriggerEvent):
        """Activate a trigger and request reprediction"""
        try:
            self.active_triggers[trigger.trigger_id] = trigger
            self.trigger_history.append(trigger)
            
            # Update statistics
            self.trigger_stats['total_triggers'] += 1
            self.trigger_stats['triggers_by_type'][trigger.trigger_type.value] = \
                self.trigger_stats['triggers_by_type'].get(trigger.trigger_type.value, 0) + 1
            self.trigger_stats['triggers_by_priority'][trigger.priority.value] = \
                self.trigger_stats['triggers_by_priority'].get(trigger.priority.value, 0) + 1
            
            # Request reprediction based on priority
            self._request_reprediction(trigger)
            
            # Call trigger activated callback
            if self.on_trigger_activated:
                self.on_trigger_activated(trigger)
            
            logger.info(f"Trigger activated: {trigger.trigger_id} for {trigger.symbol}")
            
        except Exception as e:
            logger.error(f"Error activating trigger {trigger.trigger_id}: {e}")
    
    def _request_reprediction(self, trigger: TriggerEvent):
        """Request reprediction based on trigger priority"""
        try:
            # Determine update interval based on priority
            if trigger.priority == TriggerPriority.CRITICAL:
                # Force immediate update
                task_id = self.update_scheduler.force_update(trigger.symbol, priority=1)
            elif trigger.priority == TriggerPriority.HIGH:
                # Schedule urgent update
                task_id = self.update_scheduler.force_update(trigger.symbol, priority=2)
            else:
                # Schedule normal update
                task_id = self.update_scheduler.force_update(trigger.symbol, priority=3)
            
            # Mark trigger as processed
            trigger.processed = True
            trigger.processed_at = datetime.now()
            self.trigger_stats['processed_triggers'] += 1
            
            # Call reprediction requested callback
            if self.on_reprediction_requested:
                self.on_reprediction_requested(trigger, task_id)
            
            logger.info(f"Reprediction requested for {trigger.symbol} due to {trigger.trigger_type.value}")
            
        except Exception as e:
            logger.error(f"Error requesting reprediction for trigger {trigger.trigger_id}: {e}")
            self.trigger_stats['failed_triggers'] += 1
    
    def _get_recent_model_accuracy(self, symbol: str, days_back: int = 7) -> Optional[float]:
        """Get recent model accuracy for a symbol"""
        try:
            # This would typically query the database for recent prediction accuracy
            # For now, return a placeholder
            return 0.72  # Placeholder accuracy
            
        except Exception as e:
            logger.error(f"Error getting recent accuracy for {symbol}: {e}")
            return None
    
    def _get_baseline_model_accuracy(self, symbol: str) -> Optional[float]:
        """Get baseline model accuracy from training"""
        try:
            # This would typically query the database for baseline accuracy
            # For now, return a placeholder
            return 0.85  # Placeholder baseline accuracy
            
        except Exception as e:
            logger.error(f"Error getting baseline accuracy for {symbol}: {e}")
            return None
    
    def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get trigger statistics"""
        return {
            'total_triggers': self.trigger_stats['total_triggers'],
            'processed_triggers': self.trigger_stats['processed_triggers'],
            'failed_triggers': self.trigger_stats['failed_triggers'],
            'success_rate': (self.trigger_stats['processed_triggers'] / 
                           max(1, self.trigger_stats['total_triggers'])) * 100,
            'triggers_by_type': self.trigger_stats['triggers_by_type'],
            'triggers_by_priority': self.trigger_stats['triggers_by_priority'],
            'active_triggers': len(self.active_triggers),
            'recent_triggers': [
                {
                    'trigger_id': trigger.trigger_id,
                    'symbol': trigger.symbol,
                    'type': trigger.trigger_type.value,
                    'priority': trigger.priority.value,
                    'timestamp': trigger.timestamp.isoformat(),
                    'reason': trigger.reason
                }
                for trigger in self.trigger_history[-10:]  # Last 10 triggers
            ]
        }
    
    def cleanup_old_triggers(self, hours_back: int = 24):
        """Clean up old processed triggers"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Remove old active triggers
            old_active = [
                trigger_id for trigger_id, trigger in self.active_triggers.items()
                if trigger.timestamp < cutoff_time
            ]
            for trigger_id in old_active:
                del self.active_triggers[trigger_id]
            
            # Keep only recent trigger history
            self.trigger_history = [
                trigger for trigger in self.trigger_history
                if trigger.timestamp > cutoff_time
            ]
            
            logger.info(f"Cleaned up {len(old_active)} old triggers")
            
        except Exception as e:
            logger.error(f"Error cleaning up old triggers: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("RepredictionTriggers module loaded successfully")
    print("Use with DataStorage, ValidationWorkflow, and UpdateScheduler instances for full functionality")
