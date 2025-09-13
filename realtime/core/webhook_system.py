"""
Webhook Notification System for Real-time LSTM Prediction System
Phase 4.2: Real-time webhook notifications for external integrations
"""

import json
import logging
import requests
import threading
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from urllib.parse import urljoin
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class WebhookEventType(Enum):
    """Webhook event types"""
    PREDICTION_UPDATE = "prediction_update"
    TRADE_RECOMMENDATION = "trade_recommendation"
    PERFORMANCE_ALERT = "performance_alert"
    SYSTEM_STATUS = "system_status"
    VALIDATION_RESULT = "validation_result"
    ERROR_EVENT = "error_event"
    MODEL_UPDATE = "model_update"
    DATA_QUALITY_ALERT = "data_quality_alert"

class WebhookStatus(Enum):
    """Webhook status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    SUSPENDED = "suspended"

@dataclass
class Webhook:
    """Webhook configuration"""
    webhook_id: str
    name: str
    url: str
    event_types: List[WebhookEventType]
    secret_key: Optional[str] = None
    status: WebhookStatus = WebhookStatus.ACTIVE
    retry_count: int = 3
    timeout_seconds: int = 30
    headers: Dict[str, str] = None
    filters: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    last_triggered: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    last_error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.headers is None:
            self.headers = {'Content-Type': 'application/json'}
        if self.filters is None:
            self.filters = {}

@dataclass
class WebhookPayload:
    """Webhook payload structure"""
    event_type: str
    event_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class WebhookDelivery:
    """Webhook delivery record"""
    delivery_id: str
    webhook_id: str
    payload: WebhookPayload
    status: str
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    delivery_time: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None

class WebhookDeliverer:
    """Handle webhook delivery"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.get('default_timeout', 30)
        
        # Delivery queue
        self.delivery_queue = queue.Queue(maxsize=config.get('delivery_queue_size', 1000))
        self.running = False
        self.delivery_threads = []
        
        # Statistics
        self.stats = {
            'total_deliveries': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'retry_count': 0,
            'avg_delivery_time_ms': 0
        }
    
    def start(self):
        """Start webhook delivery workers"""
        if self.running:
            return
        
        self.running = True
        worker_count = self.config.get('delivery_workers', 5)
        
        for i in range(worker_count):
            thread = threading.Thread(target=self._delivery_worker, daemon=True)
            thread.start()
            self.delivery_threads.append(thread)
        
        logger.info(f"Started {worker_count} webhook delivery workers")
    
    def stop(self):
        """Stop webhook delivery workers"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.delivery_threads:
            thread.join(timeout=10)
        
        logger.info("Stopped webhook delivery workers")
    
    def deliver_webhook(self, webhook: Webhook, payload: WebhookPayload):
        """Queue webhook for delivery"""
        try:
            delivery = WebhookDelivery(
                delivery_id=str(uuid.uuid4()),
                webhook_id=webhook.webhook_id,
                payload=payload,
                status='queued'
            )
            
            self.delivery_queue.put(delivery)
            logger.debug(f"Queued webhook delivery: {delivery.delivery_id}")
            
        except Exception as e:
            logger.error(f"Error queuing webhook delivery: {e}")
    
    def _delivery_worker(self):
        """Worker thread for webhook delivery"""
        while self.running:
            try:
                # Get delivery from queue
                delivery = self.delivery_queue.get(timeout=1)
                
                # Get webhook configuration
                webhook = self._get_webhook(delivery.webhook_id)
                if not webhook:
                    logger.error(f"Webhook not found: {delivery.webhook_id}")
                    continue
                
                # Deliver webhook
                self._deliver_webhook(webhook, delivery)
                
                self.delivery_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in delivery worker: {e}")
    
    def _get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Get webhook configuration (placeholder - should be implemented)"""
        # This should be implemented to get webhook from database
        return None
    
    def _deliver_webhook(self, webhook: Webhook, delivery: WebhookDelivery):
        """Deliver webhook to target URL"""
        start_time = time.time()
        
        try:
            # Prepare payload
            payload_json = json.dumps(asdict(delivery.payload), default=str)
            
            # Add signature if secret key is provided
            headers = webhook.headers.copy()
            if webhook.secret_key:
                signature = self._generate_signature(payload_json, webhook.secret_key)
                headers['X-Webhook-Signature'] = signature
            
            # Add delivery metadata
            headers['X-Webhook-Event'] = delivery.payload.event_type
            headers['X-Webhook-Delivery-ID'] = delivery.delivery_id
            headers['X-Webhook-Timestamp'] = delivery.payload.timestamp.isoformat()
            
            # Make request
            response = self.session.post(
                webhook.url,
                data=payload_json,
                headers=headers,
                timeout=webhook.timeout_seconds
            )
            
            # Update delivery record
            delivery.status = 'delivered'
            delivery.response_code = response.status_code
            delivery.response_body = response.text
            delivery.delivery_time = datetime.now()
            
            # Update webhook statistics
            if response.status_code < 400:
                webhook.success_count += 1
                webhook.last_triggered = datetime.now()
                self.stats['successful_deliveries'] += 1
            else:
                webhook.failure_count += 1
                webhook.last_error = f"HTTP {response.status_code}: {response.text}"
                self.stats['failed_deliveries'] += 1
                delivery.status = 'failed'
                delivery.error_message = webhook.last_error
            
            # Store delivery record
            self._store_delivery_record(delivery)
            
        except requests.exceptions.Timeout:
            delivery.status = 'timeout'
            delivery.error_message = f"Timeout after {webhook.timeout_seconds} seconds"
            webhook.failure_count += 1
            self.stats['failed_deliveries'] += 1
            
        except requests.exceptions.ConnectionError as e:
            delivery.status = 'connection_error'
            delivery.error_message = str(e)
            webhook.failure_count += 1
            webhook.last_error = str(e)
            self.stats['failed_deliveries'] += 1
            
        except Exception as e:
            delivery.status = 'error'
            delivery.error_message = str(e)
            webhook.failure_count += 1
            webhook.last_error = str(e)
            self.stats['failed_deliveries'] += 1
            
        finally:
            delivery.delivery_time = datetime.now()
            delivery_time_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self.stats['total_deliveries'] += 1
            self._update_avg_delivery_time(delivery_time_ms)
            
            # Store delivery record
            self._store_delivery_record(delivery)
            
            logger.debug(f"Webhook delivery completed: {delivery.delivery_id} - {delivery.status}")
    
    def _generate_signature(self, payload: str, secret_key: str) -> str:
        """Generate HMAC signature for webhook payload"""
        return hmac.new(
            secret_key.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _store_delivery_record(self, delivery: WebhookDelivery):
        """Store webhook delivery record (placeholder - should be implemented)"""
        # This should be implemented to store delivery records in database
        pass
    
    def _update_avg_delivery_time(self, delivery_time_ms: float):
        """Update average delivery time"""
        current_avg = self.stats['avg_delivery_time_ms']
        total_deliveries = self.stats['total_deliveries']
        
        if total_deliveries == 1:
            self.stats['avg_delivery_time_ms'] = delivery_time_ms
        else:
            # Calculate rolling average
            self.stats['avg_delivery_time_ms'] = (
                (current_avg * (total_deliveries - 1) + delivery_time_ms) / total_deliveries
            )

class WebhookSystem:
    """Main webhook notification system"""
    
    def __init__(self, data_storage, config: Dict[str, Any]):
        self.data_storage = data_storage
        self.config = config
        
        # Webhook storage
        self.webhooks: Dict[str, Webhook] = {}
        
        # Webhook deliverer
        self.deliverer = WebhookDeliverer(config.get('delivery', {}))
        
        # Event filters
        self.event_filters = {}
        
        # Load existing webhooks
        self._load_webhooks()
        
        # Start delivery system
        self.deliverer.start()
        
        logger.info("WebhookSystem initialized")
    
    def _load_webhooks(self):
        """Load webhooks from database"""
        try:
            # This should be implemented to load webhooks from database
            # For now, we'll use an empty dictionary
            self.webhooks = {}
            logger.info("Loaded webhooks from database")
            
        except Exception as e:
            logger.error(f"Error loading webhooks: {e}")
    
    def create_webhook(self, name: str, url: str, event_types: List[WebhookEventType],
                      secret_key: str = None, headers: Dict[str, str] = None,
                      filters: Dict[str, Any] = None) -> str:
        """Create a new webhook"""
        try:
            webhook_id = str(uuid.uuid4())
            
            webhook = Webhook(
                webhook_id=webhook_id,
                name=name,
                url=url,
                event_types=event_types,
                secret_key=secret_key,
                headers=headers,
                filters=filters or {}
            )
            
            # Store webhook
            self.webhooks[webhook_id] = webhook
            self._store_webhook(webhook)
            
            logger.info(f"Created webhook: {webhook_id} - {name}")
            return webhook_id
            
        except Exception as e:
            logger.error(f"Error creating webhook: {e}")
            raise
    
    def update_webhook(self, webhook_id: str, **updates) -> bool:
        """Update webhook configuration"""
        try:
            if webhook_id not in self.webhooks:
                return False
            
            webhook = self.webhooks[webhook_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(webhook, key):
                    setattr(webhook, key, value)
            
            webhook.updated_at = datetime.now()
            
            # Store updated webhook
            self._store_webhook(webhook)
            
            logger.info(f"Updated webhook: {webhook_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating webhook: {e}")
            return False
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete webhook"""
        try:
            if webhook_id not in self.webhooks:
                return False
            
            del self.webhooks[webhook_id]
            self._delete_webhook(webhook_id)
            
            logger.info(f"Deleted webhook: {webhook_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting webhook: {e}")
            return False
    
    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Get webhook by ID"""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self, status: WebhookStatus = None) -> List[Webhook]:
        """List webhooks"""
        webhooks = list(self.webhooks.values())
        
        if status:
            webhooks = [w for w in webhooks if w.status == status]
        
        return webhooks
    
    def trigger_webhook(self, event_type: WebhookEventType, data: Dict[str, Any],
                       metadata: Dict[str, Any] = None, filters: Dict[str, Any] = None):
        """Trigger webhooks for an event"""
        try:
            # Create payload
            payload = WebhookPayload(
                event_type=event_type.value,
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                data=data,
                metadata=metadata or {}
            )
            
            # Find matching webhooks
            matching_webhooks = self._find_matching_webhooks(event_type, filters)
            
            # Deliver to matching webhooks
            for webhook in matching_webhooks:
                self.deliverer.deliver_webhook(webhook, payload)
            
            logger.debug(f"Triggered {len(matching_webhooks)} webhooks for event: {event_type.value}")
            
        except Exception as e:
            logger.error(f"Error triggering webhooks: {e}")
    
    def _find_matching_webhooks(self, event_type: WebhookEventType, 
                               filters: Dict[str, Any] = None) -> List[Webhook]:
        """Find webhooks that match the event type and filters"""
        matching_webhooks = []
        
        for webhook in self.webhooks.values():
            # Check if webhook is active
            if webhook.status != WebhookStatus.ACTIVE:
                continue
            
            # Check if webhook handles this event type
            if event_type not in webhook.event_types:
                continue
            
            # Check filters
            if filters and webhook.filters:
                if not self._filters_match(filters, webhook.filters):
                    continue
            
            matching_webhooks.append(webhook)
        
        return matching_webhooks
    
    def _filters_match(self, event_filters: Dict[str, Any], 
                      webhook_filters: Dict[str, Any]) -> bool:
        """Check if event filters match webhook filters"""
        for key, webhook_value in webhook_filters.items():
            if key not in event_filters:
                return False
            
            event_value = event_filters[key]
            
            # Handle different filter types
            if isinstance(webhook_value, list):
                # List filter - event value must be in list
                if event_value not in webhook_value:
                    return False
            elif isinstance(webhook_value, dict):
                # Range filter
                if 'min' in webhook_value and event_value < webhook_value['min']:
                    return False
                if 'max' in webhook_value and event_value > webhook_value['max']:
                    return False
            else:
                # Exact match
                if event_value != webhook_value:
                    return False
        
        return True
    
    def _store_webhook(self, webhook: Webhook):
        """Store webhook in database (placeholder - should be implemented)"""
        # This should be implemented to store webhook in database
        pass
    
    def _delete_webhook(self, webhook_id: str):
        """Delete webhook from database (placeholder - should be implemented)"""
        # This should be implemented to delete webhook from database
        pass
    
    def get_webhook_statistics(self) -> Dict[str, Any]:
        """Get webhook statistics"""
        total_webhooks = len(self.webhooks)
        active_webhooks = len([w for w in self.webhooks.values() if w.status == WebhookStatus.ACTIVE])
        
        total_successes = sum(w.success_count for w in self.webhooks.values())
        total_failures = sum(w.failure_count for w in self.webhooks.values())
        
        return {
            'total_webhooks': total_webhooks,
            'active_webhooks': active_webhooks,
            'inactive_webhooks': total_webhooks - active_webhooks,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'success_rate': total_successes / max(1, total_successes + total_failures),
            'delivery_stats': self.deliverer.stats
        }
    
    def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Test webhook delivery"""
        try:
            webhook = self.get_webhook(webhook_id)
            if not webhook:
                return {'error': 'Webhook not found'}
            
            # Create test payload
            test_payload = WebhookPayload(
                event_type='test',
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                data={'test': True, 'message': 'This is a test webhook delivery'},
                metadata={'webhook_id': webhook_id}
            )
            
            # Deliver test webhook
            self.deliverer.deliver_webhook(webhook, test_payload)
            
            return {
                'status': 'test_delivered',
                'webhook_id': webhook_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error testing webhook: {e}")
            return {'error': str(e)}
    
    def stop(self):
        """Stop webhook system"""
        try:
            self.deliverer.stop()
            logger.info("WebhookSystem stopped")
        except Exception as e:
            logger.error(f"Error stopping webhook system: {e}")

# Convenience functions for triggering webhooks
def trigger_prediction_webhook(webhook_system: WebhookSystem, symbol: str, 
                              prediction_data: Dict[str, Any]):
    """Trigger webhook for prediction update"""
    webhook_system.trigger_webhook(
        WebhookEventType.PREDICTION_UPDATE,
        prediction_data,
        metadata={'symbol': symbol}
    )

def trigger_recommendation_webhook(webhook_system: WebhookSystem, symbol: str,
                                  recommendation_data: Dict[str, Any]):
    """Trigger webhook for trade recommendation"""
    webhook_system.trigger_webhook(
        WebhookEventType.TRADE_RECOMMENDATION,
        recommendation_data,
        metadata={'symbol': symbol}
    )

def trigger_alert_webhook(webhook_system: WebhookSystem, alert_data: Dict[str, Any]):
    """Trigger webhook for performance alert"""
    webhook_system.trigger_webhook(
        WebhookEventType.PERFORMANCE_ALERT,
        alert_data
    )

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("WebhookSystem module loaded successfully")
    print("Use with DataStorage instance for full functionality")
