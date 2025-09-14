"""
REST API Server for Real-time LSTM Prediction System
Phase 2.4: REST API interface for external access and integration
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import time

from .realtime_system import RealTimeSystem
from .config import Config
from monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class APIServer:
    """
    REST API server for the real-time prediction system
    """
    
    def __init__(self, realtime_system: RealTimeSystem, config: Dict[str, Any]):
        """
        Initialize the API server
        
        Args:
            realtime_system: RealTimeSystem instance
            config: Configuration dictionary
        """
        self.realtime_system = realtime_system
        self.config = config
        self.data_storage = realtime_system.data_storage
        
        # Flask app setup
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for cross-origin requests
        
        # API configuration
        self.api_config = config.get('api', {})
        self.host = self.api_config.get('host', '0.0.0.0')
        self.port = self.api_config.get('port', 5000)
        self.debug = self.api_config.get('debug', False)
        
        # Enhanced Authentication
        self.api_keys = self.api_config.get('api_keys', {'default': 'default_api_key'})
        self.require_auth = self.api_config.get('require_auth', True)
        self.jwt_secret = self.api_config.get('jwt_secret', 'your-secret-key')
        
        # Enhanced Rate limiting with different tiers
        self.rate_limits = {}
        self.rate_limit_config = self.api_config.get('rate_limits', {
            'default': {'window': 60, 'requests': 100},
            'premium': {'window': 60, 'requests': 1000},
            'admin': {'window': 60, 'requests': 10000}
        })
        
        # Setup routes
        self._setup_routes()
        
        # Setup performance monitoring
        self.performance_monitor = PerformanceMonitor(self.data_storage, config.get('monitoring', {}))
        
        logger.info(f"APIServer initialized on {self.host}:{self.port}")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return self._handle_health_check()
        
        # System status
        @self.app.route('/status', methods=['GET'])
        def system_status():
            return self._handle_system_status()
        
        # Predictions endpoints
        @self.app.route('/predictions', methods=['GET'])
        def get_predictions():
            return self._handle_get_predictions()
        
        @self.app.route('/predictions/<symbol>', methods=['GET'])
        def get_symbol_predictions(symbol):
            return self._handle_get_symbol_predictions(symbol)
        
        # Trade recommendations
        @self.app.route('/recommendations', methods=['GET'])
        def get_recommendations():
            return self._handle_get_recommendations()
        
        @self.app.route('/recommendations/<symbol>', methods=['GET'])
        def get_symbol_recommendations(symbol):
            return self._handle_get_symbol_recommendations(symbol)
        
        # Performance monitoring
        @self.app.route('/performance', methods=['GET'])
        def get_performance():
            return self._handle_get_performance()
        
        @self.app.route('/performance/<symbol>', methods=['GET'])
        def get_symbol_performance(symbol):
            return self._handle_get_symbol_performance(symbol)
        
        @self.app.route('/alerts', methods=['GET'])
        def get_alerts():
            return self._handle_get_alerts()
        
        # System control
        @self.app.route('/control/update', methods=['POST'])
        def force_update():
            return self._handle_force_update()
        
        @self.app.route('/control/update/<symbol>', methods=['POST'])
        def force_update_symbol(symbol):
            return self._handle_force_update_symbol(symbol)
        
        # Configuration
        @self.app.route('/config', methods=['GET'])
        def get_config():
            return self._handle_get_config()
        
        @self.app.route('/config', methods=['POST'])
        def update_config():
            return self._handle_update_config()
        
        # Historical Performance
        @self.app.route('/historical/performance', methods=['GET'])
        def get_historical_performance():
            return self._handle_get_historical_performance()
        
        @self.app.route('/historical/predictions', methods=['GET'])
        def get_historical_predictions():
            return self._handle_get_historical_predictions()
        
        # Data Export
        @self.app.route('/export/csv', methods=['POST'])
        def export_csv():
            return self._handle_export_csv()
        
        @self.app.route('/export/json', methods=['POST'])
        def export_json():
            return self._handle_export_json()
        
        @self.app.route('/export/status/<export_id>', methods=['GET'])
        def get_export_status(export_id):
            return self._handle_get_export_status(export_id)
        
        # Real-time Streaming
        @self.app.route('/stream/predictions', methods=['GET'])
        def stream_predictions():
            return self._handle_stream_predictions()
        
        @self.app.route('/stream/alerts', methods=['GET'])
        def stream_alerts():
            return self._handle_stream_alerts()
        
        # Webhook Management
        @self.app.route('/webhooks', methods=['GET'])
        def get_webhooks():
            return self._handle_get_webhooks()
        
        @self.app.route('/webhooks', methods=['POST'])
        def create_webhook():
            return self._handle_create_webhook()
        
        @self.app.route('/webhooks/<webhook_id>', methods=['PUT'])
        def update_webhook(webhook_id):
            return self._handle_update_webhook(webhook_id)
        
        @self.app.route('/webhooks/<webhook_id>', methods=['DELETE'])
        def delete_webhook(webhook_id):
            return self._handle_delete_webhook(webhook_id)
        
        # Data Visualization
        @self.app.route('/visualization/performance', methods=['GET'])
        def get_performance_chart():
            return self._handle_get_performance_chart()
        
        @self.app.route('/visualization/predictions', methods=['GET'])
        def get_predictions_chart():
            return self._handle_get_predictions_chart()
        
        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def _authenticate_request(self) -> tuple[bool, str]:
        """Authenticate API request and return (success, tier)"""
        if not self.require_auth:
            return True, 'default'
        
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not api_key:
            return False, 'none'
        
        # Check API key and determine tier
        for tier, key in self.api_keys.items():
            if api_key == key:
                return True, tier
        
        return False, 'none'
    
    def _rate_limit_check(self, client_ip: str, tier: str = 'default') -> bool:
        """Check rate limiting for client based on tier"""
        current_time = time.time()
        
        # Get rate limit config for tier
        tier_config = self.rate_limit_config.get(tier, self.rate_limit_config['default'])
        window = tier_config['window']
        max_requests = tier_config['requests']
        
        client_key = f"{client_ip}_{tier}"
        if client_key not in self.rate_limits:
            self.rate_limits[client_key] = []
        
        # Remove old requests outside the window
        self.rate_limits[client_key] = [
            req_time for req_time in self.rate_limits[client_key]
            if current_time - req_time < window
        ]
        
        # Check if under limit
        if len(self.rate_limits[client_key]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limits[client_key].append(current_time)
        return True
    
    def _handle_health_check(self):
        """Handle health check endpoint"""
        try:
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0',
                'uptime': 'N/A'  # Could track actual uptime
            })
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_system_status(self):
        """Handle system status endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            status = self.realtime_system.get_system_status()
            
            # Add API-specific status
            status['api'] = {
                'host': self.host,
                'port': self.port,
                'rate_limiting': True,
                'authentication': self.require_auth
            }
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_predictions(self):
        """Handle get all predictions endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            predictions = self.realtime_system.get_current_predictions()
            
            # Format predictions for API response
            formatted_predictions = {}
            for symbol, pred_data in predictions.items():
                formatted_predictions[symbol] = {
                    'symbol': symbol,
                    'timestamp': pred_data['timestamp'].isoformat(),
                    'confidence_score': pred_data['confidence_score'],
                    'total_predictions': pred_data['total_predictions'],
                    'prediction_horizon': pred_data['prediction_horizon'],
                    'latest_predictions': pred_data['predictions'].tail(10).to_dict('records') if not pred_data['predictions'].empty else []
                }
            
            return jsonify({
                'predictions': formatted_predictions,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_symbol_predictions(self, symbol):
        """Handle get symbol predictions endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            predictions = self.realtime_system.get_current_predictions(symbol)
            
            if not predictions:
                return jsonify({'error': f'No predictions available for {symbol}'}), 404
            
            # Format prediction data
            formatted_prediction = {
                'symbol': symbol,
                'timestamp': predictions['timestamp'].isoformat(),
                'confidence_score': predictions['confidence_score'],
                'total_predictions': predictions['total_predictions'],
                'prediction_horizon': predictions['prediction_horizon'],
                'predictions': predictions['predictions'].to_dict('records') if not predictions['predictions'].empty else []
            }
            
            return jsonify(formatted_prediction)
            
        except Exception as e:
            logger.error(f"Error getting predictions for {symbol}: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_recommendations(self):
        """Handle get all recommendations endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            recommendations = self.realtime_system.get_trade_recommendations()
            
            return jsonify({
                'recommendations': recommendations,
                'total_count': len(recommendations),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_symbol_recommendations(self, symbol):
        """Handle get symbol recommendations endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            recommendations = self.realtime_system.get_trade_recommendations(symbol)
            
            return jsonify({
                'symbol': symbol,
                'recommendations': recommendations,
                'total_count': len(recommendations),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting recommendations for {symbol}: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_performance(self):
        """Handle get system performance endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get performance summary
            summary = self.performance_monitor.get_metrics_summary()
            
            # Get active alerts
            active_alerts = self.performance_monitor.get_active_alerts()
            formatted_alerts = []
            for alert_key, alert in active_alerts.items():
                formatted_alerts.append({
                    'key': alert_key,
                    'level': alert.level.value,
                    'metric_type': alert.metric_type.value,
                    'symbol': alert.symbol,
                    'message': alert.message,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'timestamp': alert.timestamp.isoformat()
                })
            
            return jsonify({
                'performance_summary': summary,
                'active_alerts': formatted_alerts,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_symbol_performance(self, symbol):
        """Handle get symbol performance endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get performance report
            period_hours = int(request.args.get('period_hours', 24))
            report = self.performance_monitor.get_performance_report(symbol, period_hours)
            
            # Format report
            formatted_report = {
                'symbol': report.symbol,
                'period_hours': report.period_hours,
                'overall_score': report.overall_score,
                'metrics': report.metrics,
                'alerts': [{
                    'level': alert.level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                } for alert in report.alerts],
                'recommendations': report.recommendations,
                'timestamp': report.timestamp.isoformat()
            }
            
            return jsonify(formatted_report)
            
        except Exception as e:
            logger.error(f"Error getting performance for {symbol}: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_alerts(self):
        """Handle get alerts endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get query parameters
            level = request.args.get('level')
            symbol = request.args.get('symbol')
            limit = int(request.args.get('limit', 50))
            
            # Get active alerts
            active_alerts = self.performance_monitor.get_active_alerts()
            
            # Filter alerts
            filtered_alerts = []
            for alert_key, alert in active_alerts.items():
                if level and alert.level.value != level:
                    continue
                if symbol and alert.symbol != symbol:
                    continue
                
                filtered_alerts.append({
                    'key': alert_key,
                    'level': alert.level.value,
                    'metric_type': alert.metric_type.value,
                    'symbol': alert.symbol,
                    'message': alert.message,
                    'value': alert.value,
                    'threshold': alert.threshold,
                    'timestamp': alert.timestamp.isoformat()
                })
            
            # Limit results
            filtered_alerts = filtered_alerts[:limit]
            
            return jsonify({
                'alerts': filtered_alerts,
                'total_count': len(filtered_alerts),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_force_update(self):
        """Handle force update endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Force update all symbols
            self.realtime_system.force_update()
            
            return jsonify({
                'message': 'Update triggered for all symbols',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in force update: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_force_update_symbol(self, symbol):
        """Handle force update symbol endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Force update specific symbol
            self.realtime_system.force_update(symbol)
            
            return jsonify({
                'message': f'Update triggered for {symbol}',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in force update for {symbol}: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_config(self):
        """Handle get configuration endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get system configuration (without sensitive data)
            config = self.realtime_system.config.copy()
            
            # Remove sensitive information
            sensitive_keys = ['polygon_api_key', 'api_key', 'password']
            for key in sensitive_keys:
                if key in config:
                    config[key] = '***'
            
            return jsonify({
                'configuration': config,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting configuration: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_update_config(self):
        """Handle update configuration endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get configuration updates from request
            updates = request.get_json()
            
            if not updates:
                return jsonify({'error': 'No configuration updates provided'}), 400
            
            # Validate and apply updates
            # This is a simplified implementation - in production you'd want more validation
            allowed_keys = ['min_profit_percent', 'update_interval', 'confidence_threshold']
            
            applied_updates = {}
            for key, value in updates.items():
                if key in allowed_keys:
                    self.realtime_system.config[key] = value
                    applied_updates[key] = value
            
            return jsonify({
                'message': 'Configuration updated',
                'applied_updates': applied_updates,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return jsonify({'error': str(e)}), 500
    
    # New handler methods for enhanced functionality
    def _handle_get_historical_performance(self):
        """Handle get historical performance endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get query parameters
            symbol = request.args.get('symbol')
            days_back = int(request.args.get('days_back', 30))
            
            # Get historical performance data
            performance_data = self.realtime_system.get_historical_performance(symbol, days_back)
            
            return jsonify({
                'performance_data': performance_data,
                'symbol': symbol,
                'days_back': days_back,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_historical_predictions(self):
        """Handle get historical predictions endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get query parameters
            symbol = request.args.get('symbol')
            days_back = int(request.args.get('days_back', 7))
            limit = int(request.args.get('limit', 100))
            
            # Get historical predictions
            predictions = self.realtime_system.get_historical_predictions(symbol, days_back, limit)
            
            return jsonify({
                'predictions': predictions,
                'symbol': symbol,
                'days_back': days_back,
                'limit': limit,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting historical predictions: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_export_csv(self):
        """Handle CSV export endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get export parameters
            export_data = request.get_json()
            if not export_data:
                return jsonify({'error': 'Export parameters required'}), 400
            
            # Create export request
            export_id = self.realtime_system.create_export_request(
                export_type='csv',
                **export_data
            )
            
            return jsonify({
                'export_id': export_id,
                'status': 'initiated',
                'message': 'CSV export request created',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating CSV export: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_export_json(self):
        """Handle JSON export endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get export parameters
            export_data = request.get_json()
            if not export_data:
                return jsonify({'error': 'Export parameters required'}), 400
            
            # Create export request
            export_id = self.realtime_system.create_export_request(
                export_type='json',
                **export_data
            )
            
            return jsonify({
                'export_id': export_id,
                'status': 'initiated',
                'message': 'JSON export request created',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating JSON export: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_export_status(self, export_id):
        """Handle get export status endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get export status
            status = self.realtime_system.get_export_status(export_id)
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Error getting export status: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_stream_predictions(self):
        """Handle real-time prediction streaming endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            # Streaming endpoints have different rate limits
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Create streaming response
            def generate():
                while True:
                    predictions = self.realtime_system.get_current_predictions()
                    yield f"data: {json.dumps(predictions)}\n\n"
                    time.sleep(5)  # Update every 5 seconds
            
            return Response(generate(), mimetype='text/event-stream')
            
        except Exception as e:
            logger.error(f"Error in prediction streaming: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_stream_alerts(self):
        """Handle real-time alert streaming endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Create streaming response
            def generate():
                while True:
                    alerts = self.performance_monitor.get_active_alerts()
                    yield f"data: {json.dumps(alerts)}\n\n"
                    time.sleep(10)  # Update every 10 seconds
            
            return Response(generate(), mimetype='text/event-stream')
            
        except Exception as e:
            logger.error(f"Error in alert streaming: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_webhooks(self):
        """Handle get webhooks endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get webhooks
            webhooks = self.realtime_system.get_webhooks()
            
            return jsonify({
                'webhooks': webhooks,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting webhooks: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_create_webhook(self):
        """Handle create webhook endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get webhook data
            webhook_data = request.get_json()
            if not webhook_data:
                return jsonify({'error': 'Webhook data required'}), 400
            
            # Create webhook
            webhook_id = self.realtime_system.create_webhook(**webhook_data)
            
            return jsonify({
                'webhook_id': webhook_id,
                'status': 'created',
                'message': 'Webhook created successfully',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating webhook: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_update_webhook(self, webhook_id):
        """Handle update webhook endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get update data
            update_data = request.get_json()
            if not update_data:
                return jsonify({'error': 'Update data required'}), 400
            
            # Update webhook
            success = self.realtime_system.update_webhook(webhook_id, **update_data)
            
            if success:
                return jsonify({
                    'webhook_id': webhook_id,
                    'status': 'updated',
                    'message': 'Webhook updated successfully',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Webhook not found'}), 404
            
        except Exception as e:
            logger.error(f"Error updating webhook: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_delete_webhook(self, webhook_id):
        """Handle delete webhook endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Delete webhook
            success = self.realtime_system.delete_webhook(webhook_id)
            
            if success:
                return jsonify({
                    'webhook_id': webhook_id,
                    'status': 'deleted',
                    'message': 'Webhook deleted successfully',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Webhook not found'}), 404
            
        except Exception as e:
            logger.error(f"Error deleting webhook: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_performance_chart(self):
        """Handle get performance chart data endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get chart parameters
            symbol = request.args.get('symbol')
            hours_back = int(request.args.get('hours_back', 24))
            chart_type = request.args.get('chart_type', 'line')
            
            # Get chart data
            chart_data = self.realtime_system.get_performance_chart_data(symbol, hours_back, chart_type)
            
            return jsonify({
                'chart_data': chart_data,
                'symbol': symbol,
                'hours_back': hours_back,
                'chart_type': chart_type,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting performance chart: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_get_predictions_chart(self):
        """Handle get predictions chart data endpoint"""
        try:
            auth_success, tier = self._authenticate_request()
            if not auth_success:
                return jsonify({'error': 'Authentication required'}), 401
            
            if not self._rate_limit_check(request.remote_addr, tier):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            
            # Get chart parameters
            symbol = request.args.get('symbol')
            days_back = int(request.args.get('days_back', 7))
            chart_type = request.args.get('chart_type', 'candlestick')
            
            # Get chart data
            chart_data = self.realtime_system.get_predictions_chart_data(symbol, days_back, chart_type)
            
            return jsonify({
                'chart_data': chart_data,
                'symbol': symbol,
                'days_back': days_back,
                'chart_type': chart_type,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting predictions chart: {e}")
            return jsonify({'error': str(e)}), 500
    
    def start(self):
        """Start the API server"""
        try:
            logger.info(f"Starting API server on {self.host}:{self.port}")
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            raise
    
    def stop(self):
        """Stop the API server"""
        try:
            self.performance_monitor.stop_monitoring()
            logger.info("API server stopped")
        except Exception as e:
            logger.error(f"Error stopping API server: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # This would be used with actual RealTimeSystem instance
    print("API Server module loaded successfully")
    print("Use with RealTimeSystem instance for full functionality")
