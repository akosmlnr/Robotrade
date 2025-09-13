"""
Alerting System for Real-time LSTM Prediction System
Phase 3.3: Comprehensive alerting for system issues and anomalies
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import smtplib
import json
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Types of alerts"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    PREDICTION = "prediction"
    VALIDATION = "validation"
    DATA_QUALITY = "data_quality"
    MODEL = "model"
    BUSINESS = "business"
    SECURITY = "security"

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    CONSOLE = "console"
    FILE = "file"

class AlertStatus(Enum):
    """Alert status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    rule_name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # JSON string with condition logic
    channels: List[AlertChannel]
    recipients: List[str]
    cooldown_minutes: int = 15
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    status: AlertStatus = AlertStatus.PENDING
    channels_sent: List[AlertChannel] = None
    delivery_attempts: int = 0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.channels_sent is None:
            self.channels_sent = []

class AlertingSystem:
    """
    Comprehensive alerting system for system issues and anomalies
    """
    
    def __init__(self, data_storage, config: Dict[str, Any] = None):
        """
        Initialize alerting system
        
        Args:
            data_storage: DataStorage instance
            config: Optional configuration dictionary
        """
        self.data_storage = data_storage
        
        # Configuration
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        self._load_default_rules()
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        
        # Alert delivery
        self.delivery_threads: Dict[str, threading.Thread] = {}
        self.delivery_queue = []
        self.delivery_lock = threading.Lock()
        
        # Statistics
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_sent': 0,
            'alerts_failed': 0,
            'alerts_acknowledged': 0,
            'alerts_resolved': 0,
            'last_alert': None
        }
        
        # Start delivery worker
        self._start_delivery_worker()
        
        logger.info("AlertingSystem initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'email_enabled': False,
            'email_smtp_server': 'smtp.gmail.com',
            'email_smtp_port': 587,
            'email_username': '',
            'email_password': '',
            'email_from': '',
            'webhook_enabled': False,
            'webhook_url': '',
            'slack_enabled': False,
            'slack_webhook_url': '',
            'teams_enabled': False,
            'teams_webhook_url': '',
            'sms_enabled': False,
            'sms_provider': 'twilio',
            'sms_api_key': '',
            'sms_api_secret': '',
            'file_logging_enabled': True,
            'alert_log_path': 'alerts.log',
            'max_delivery_attempts': 3,
            'delivery_retry_delay_seconds': 60
        }
    
    def _load_default_rules(self):
        """Load default alert rules"""
        try:
            # System performance rules
            self._add_rule(AlertRule(
                rule_id="high_cpu_usage",
                rule_name="High CPU Usage",
                alert_type=AlertType.SYSTEM,
                severity=AlertSeverity.WARNING,
                condition='{"metric": "cpu_usage_percent", "operator": ">", "value": 80}',
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE],
                recipients=[],
                cooldown_minutes=30
            ))
            
            self._add_rule(AlertRule(
                rule_id="critical_cpu_usage",
                rule_name="Critical CPU Usage",
                alert_type=AlertType.SYSTEM,
                severity=AlertSeverity.CRITICAL,
                condition='{"metric": "cpu_usage_percent", "operator": ">", "value": 95}',
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE, AlertChannel.EMAIL],
                recipients=[],
                cooldown_minutes=15
            ))
            
            # Memory usage rules
            self._add_rule(AlertRule(
                rule_id="high_memory_usage",
                rule_name="High Memory Usage",
                alert_type=AlertType.SYSTEM,
                severity=AlertSeverity.WARNING,
                condition='{"metric": "memory_usage_percent", "operator": ">", "value": 85}',
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE],
                recipients=[],
                cooldown_minutes=30
            ))
            
            # Prediction performance rules
            self._add_rule(AlertRule(
                rule_id="low_prediction_accuracy",
                rule_name="Low Prediction Accuracy",
                alert_type=AlertType.PREDICTION,
                severity=AlertSeverity.WARNING,
                condition='{"metric": "model_accuracy", "operator": "<", "value": 0.7}',
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE, AlertChannel.EMAIL],
                recipients=[],
                cooldown_minutes=60
            ))
            
            # Validation failure rules
            self._add_rule(AlertRule(
                rule_id="high_validation_failure_rate",
                rule_name="High Validation Failure Rate",
                alert_type=AlertType.VALIDATION,
                severity=AlertSeverity.WARNING,
                condition='{"metric": "validation_failure_rate", "operator": ">", "value": 0.1}',
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE],
                recipients=[],
                cooldown_minutes=30
            ))
            
            # Data quality rules
            self._add_rule(AlertRule(
                rule_id="stale_data",
                rule_name="Stale Data",
                alert_type=AlertType.DATA_QUALITY,
                severity=AlertSeverity.WARNING,
                condition='{"metric": "data_freshness_minutes", "operator": ">", "value": 30}',
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE],
                recipients=[],
                cooldown_minutes=15
            ))
            
            # Model performance rules
            self._add_rule(AlertRule(
                rule_id="model_drift",
                rule_name="Model Drift Detected",
                alert_type=AlertType.MODEL,
                severity=AlertSeverity.CRITICAL,
                condition='{"metric": "model_drift_score", "operator": ">", "value": 0.15}',
                channels=[AlertChannel.CONSOLE, AlertChannel.FILE, AlertChannel.EMAIL],
                recipients=[],
                cooldown_minutes=120
            ))
            
            logger.info(f"Loaded {len(self.alert_rules)} default alert rules")
            
        except Exception as e:
            logger.error(f"Error loading default alert rules: {e}")
    
    def _add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.rule_id] = rule
    
    def create_alert(self, rule_id: str, title: str, message: str,
                    details: Dict[str, Any] = None) -> str:
        """
        Create an alert based on a rule
        
        Args:
            rule_id: Alert rule ID
            title: Alert title
            message: Alert message
            details: Additional alert details
            
        Returns:
            Alert ID
        """
        try:
            if rule_id not in self.alert_rules:
                logger.error(f"Alert rule {rule_id} not found")
                return ""
            
            rule = self.alert_rules[rule_id]
            if not rule.enabled:
                logger.debug(f"Alert rule {rule_id} is disabled")
                return ""
            
            # Check cooldown
            if self._is_rule_in_cooldown(rule):
                logger.debug(f"Alert rule {rule_id} is in cooldown")
                return ""
            
            # Create alert
            alert_id = f"alert_{rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule_id,
                alert_type=rule.alert_type,
                severity=rule.severity,
                title=title,
                message=message,
                details=details or {},
                timestamp=datetime.now()
            )
            
            # Add to history and active alerts
            self.alert_history.append(alert)
            self.active_alerts[alert_id] = alert
            
            # Update statistics
            self.alert_stats['total_alerts'] += 1
            self.alert_stats['last_alert'] = alert.timestamp
            
            # Queue for delivery
            self._queue_alert_for_delivery(alert)
            
            # Store in database
            self.data_storage.store_system_alert(
                alert_type=rule.alert_type.value,
                severity=rule.severity.value,
                message=message,
                details={
                    'rule_id': rule_id,
                    'rule_name': rule.rule_name,
                    'title': title,
                    'details': details or {}
                }
            )
            
            logger.info(f"Alert created: {alert_id} - {title}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return ""
    
    def _is_rule_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if a rule is in cooldown period"""
        try:
            # Find recent alerts for this rule
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.rule_id == rule.rule_id and
                (datetime.now() - alert.timestamp).total_seconds() < rule.cooldown_minutes * 60
            ]
            
            return len(recent_alerts) > 0
            
        except Exception as e:
            logger.error(f"Error checking rule cooldown: {e}")
            return False
    
    def _queue_alert_for_delivery(self, alert: Alert):
        """Queue alert for delivery"""
        try:
            with self.delivery_lock:
                self.delivery_queue.append(alert)
            
        except Exception as e:
            logger.error(f"Error queuing alert for delivery: {e}")
    
    def _start_delivery_worker(self):
        """Start alert delivery worker thread"""
        try:
            worker_thread = threading.Thread(
                target=self._delivery_worker,
                daemon=True,
                name="AlertDeliveryWorker"
            )
            worker_thread.start()
            self.delivery_threads['worker'] = worker_thread
            
        except Exception as e:
            logger.error(f"Error starting delivery worker: {e}")
    
    def _delivery_worker(self):
        """Alert delivery worker thread"""
        while True:
            try:
                # Get next alert from queue
                alert = None
                with self.delivery_lock:
                    if self.delivery_queue:
                        alert = self.delivery_queue.pop(0)
                
                if alert:
                    self._deliver_alert(alert)
                else:
                    # No alerts to deliver, wait
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in delivery worker: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _deliver_alert(self, alert: Alert):
        """Deliver alert through configured channels"""
        try:
            rule = self.alert_rules[alert.rule_id]
            
            for channel in rule.channels:
                try:
                    success = self._deliver_to_channel(alert, channel, rule.recipients)
                    
                    if success:
                        alert.channels_sent.append(channel)
                        logger.info(f"Alert {alert.alert_id} delivered via {channel.value}")
                    else:
                        logger.warning(f"Failed to deliver alert {alert.alert_id} via {channel.value}")
                
                except Exception as e:
                    logger.error(f"Error delivering alert via {channel.value}: {e}")
            
            # Update alert status
            if alert.channels_sent:
                alert.status = AlertStatus.SENT
                self.alert_stats['alerts_sent'] += 1
            else:
                alert.status = AlertStatus.FAILED
                self.alert_stats['alerts_failed'] += 1
                alert.delivery_attempts += 1
                
                # Retry if under max attempts
                if alert.delivery_attempts < self.config.get('max_delivery_attempts', 3):
                    time.sleep(self.config.get('delivery_retry_delay_seconds', 60))
                    self._queue_alert_for_delivery(alert)
            
        except Exception as e:
            logger.error(f"Error delivering alert {alert.alert_id}: {e}")
    
    def _deliver_to_channel(self, alert: Alert, channel: AlertChannel, recipients: List[str]) -> bool:
        """Deliver alert to specific channel"""
        try:
            if channel == AlertChannel.CONSOLE:
                return self._deliver_to_console(alert)
            elif channel == AlertChannel.FILE:
                return self._deliver_to_file(alert)
            elif channel == AlertChannel.EMAIL:
                return self._deliver_to_email(alert, recipients)
            elif channel == AlertChannel.WEBHOOK:
                return self._deliver_to_webhook(alert)
            elif channel == AlertChannel.SLACK:
                return self._deliver_to_slack(alert)
            elif channel == AlertChannel.TEAMS:
                return self._deliver_to_teams(alert)
            else:
                logger.warning(f"Unsupported delivery channel: {channel.value}")
                return False
                
        except Exception as e:
            logger.error(f"Error delivering to {channel.value}: {e}")
            return False
    
    def _deliver_to_console(self, alert: Alert) -> bool:
        """Deliver alert to console"""
        try:
            severity_color = {
                AlertSeverity.INFO: "\033[94m",      # Blue
                AlertSeverity.WARNING: "\033[93m",   # Yellow
                AlertSeverity.CRITICAL: "\033[91m",  # Red
                AlertSeverity.EMERGENCY: "\033[95m"  # Magenta
            }
            reset_color = "\033[0m"
            
            color = severity_color.get(alert.severity, "")
            print(f"{color}[{alert.severity.value.upper()}] {alert.title}{reset_color}")
            print(f"Time: {alert.timestamp}")
            print(f"Message: {alert.message}")
            if alert.details:
                print(f"Details: {json.dumps(alert.details, indent=2)}")
            print("-" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"Error delivering to console: {e}")
            return False
    
    def _deliver_to_file(self, alert: Alert) -> bool:
        """Deliver alert to file"""
        try:
            if not self.config.get('file_logging_enabled', True):
                return True
            
            log_path = Path(self.config.get('alert_log_path', 'alerts.log'))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, 'a') as f:
                f.write(f"[{alert.timestamp.isoformat()}] [{alert.severity.value.upper()}] {alert.title}\n")
                f.write(f"Rule: {alert.rule_id}\n")
                f.write(f"Message: {alert.message}\n")
                if alert.details:
                    f.write(f"Details: {json.dumps(alert.details)}\n")
                f.write("-" * 50 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error delivering to file: {e}")
            return False
    
    def _deliver_to_email(self, alert: Alert, recipients: List[str]) -> bool:
        """Deliver alert via email"""
        try:
            if not self.config.get('email_enabled', False) or not recipients:
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.get('email_from', '')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
            Alert: {alert.title}
            
            Severity: {alert.severity.value.upper()}
            Time: {alert.timestamp}
            Rule: {alert.rule_id}
            
            Message:
            {alert.message}
            
            Details:
            {json.dumps(alert.details, indent=2) if alert.details else 'None'}
            
            This is an automated alert from the Real-time LSTM Prediction System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config.get('email_smtp_server', ''), 
                                self.config.get('email_smtp_port', 587))
            server.starttls()
            server.login(self.config.get('email_username', ''), 
                        self.config.get('email_password', ''))
            text = msg.as_string()
            server.sendmail(self.config.get('email_from', ''), recipients, text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error delivering email: {e}")
            return False
    
    def _deliver_to_webhook(self, alert: Alert) -> bool:
        """Deliver alert via webhook"""
        try:
            if not self.config.get('webhook_enabled', False):
                return False
            
            webhook_url = self.config.get('webhook_url', '')
            if not webhook_url:
                return False
            
            payload = {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'details': alert.details
            }
            
            response = requests.post(webhook_url, json=payload, timeout=30)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error delivering webhook: {e}")
            return False
    
    def _deliver_to_slack(self, alert: Alert) -> bool:
        """Deliver alert via Slack"""
        try:
            if not self.config.get('slack_enabled', False):
                return False
            
            webhook_url = self.config.get('slack_webhook_url', '')
            if not webhook_url:
                return False
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                'attachments': [{
                    'color': color_map.get(alert.severity, 'good'),
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                        {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True},
                        {'title': 'Rule', 'value': alert.rule_id, 'short': True}
                    ],
                    'footer': 'Real-time LSTM Prediction System',
                    'ts': int(alert.timestamp.timestamp())
                }]
            }
            
            if alert.details:
                payload['attachments'][0]['fields'].append({
                    'title': 'Details',
                    'value': json.dumps(alert.details, indent=2),
                    'short': False
                })
            
            response = requests.post(webhook_url, json=payload, timeout=30)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error delivering to Slack: {e}")
            return False
    
    def _deliver_to_teams(self, alert: Alert) -> bool:
        """Deliver alert via Microsoft Teams"""
        try:
            if not self.config.get('teams_enabled', False):
                return False
            
            webhook_url = self.config.get('teams_webhook_url', '')
            if not webhook_url:
                return False
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "0078D4",
                AlertSeverity.WARNING: "FF8C00",
                AlertSeverity.CRITICAL: "D13438",
                AlertSeverity.EMERGENCY: "D13438"
            }
            
            payload = {
                '@type': 'MessageCard',
                '@context': 'http://schema.org/extensions',
                'themeColor': color_map.get(alert.severity, '0078D4'),
                'summary': alert.title,
                'sections': [{
                    'activityTitle': alert.title,
                    'activitySubtitle': f"Severity: {alert.severity.value.upper()}",
                    'activityImage': 'https://via.placeholder.com/64x64/0078D4/FFFFFF?text=!',
                    'facts': [
                        {'name': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')},
                        {'name': 'Rule', 'value': alert.rule_id}
                    ],
                    'text': alert.message
                }]
            }
            
            if alert.details:
                payload['sections'][0]['facts'].append({
                    'name': 'Details',
                    'value': json.dumps(alert.details, indent=2)
                })
            
            response = requests.post(webhook_url, json=payload, timeout=30)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error delivering to Teams: {e}")
            return False
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                alert.status = AlertStatus.ACKNOWLEDGED
                
                self.alert_stats['alerts_acknowledged'] += 1
                
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved_by = resolved_by
                alert.resolved_at = datetime.now()
                alert.status = AlertStatus.RESOLVED
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.alert_stats['alerts_resolved'] += 1
                
                logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def get_alert_status(self, alert_id: str) -> Dict[str, Any]:
        """Get alert status"""
        try:
            # Check active alerts first
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
            else:
                # Check history
                alert = None
                for a in self.alert_history:
                    if a.alert_id == alert_id:
                        alert = a
                        break
            
            if not alert:
                return {'error': f'Alert {alert_id} not found'}
            
            return {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'status': alert.status.value,
                'channels_sent': [c.value for c in alert.channels_sent],
                'delivery_attempts': alert.delivery_attempts,
                'acknowledged_by': alert.acknowledged_by,
                'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                'resolved_by': alert.resolved_by,
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                'details': alert.details
            }
            
        except Exception as e:
            logger.error(f"Error getting alert status: {e}")
            return {'error': str(e)}
    
    def get_alerting_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics"""
        try:
            return {
                'alert_stats': self.alert_stats,
                'active_alerts_count': len(self.active_alerts),
                'total_rules': len(self.alert_rules),
                'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
                'recent_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'rule_id': alert.rule_id,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'timestamp': alert.timestamp.isoformat(),
                        'status': alert.status.value
                    }
                    for alert in self.alert_history[-10:]  # Last 10 alerts
                ],
                'active_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'rule_id': alert.rule_id,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'timestamp': alert.timestamp.isoformat(),
                        'status': alert.status.value
                    }
                    for alert in self.active_alerts.values()
                ],
                'config': {
                    'email_enabled': self.config.get('email_enabled', False),
                    'webhook_enabled': self.config.get('webhook_enabled', False),
                    'slack_enabled': self.config.get('slack_enabled', False),
                    'teams_enabled': self.config.get('teams_enabled', False),
                    'file_logging_enabled': self.config.get('file_logging_enabled', True)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting alerting statistics: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("AlertingSystem module loaded successfully")
    print("Use with DataStorage instance for full functionality")
