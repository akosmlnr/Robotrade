# Quick Start Guide - Enhanced Real-time System

## 🚀 Getting Started

This guide will help you quickly set up and use the enhanced real-time LSTM prediction system with all the new features.

## Prerequisites

1. **Python 3.8+** installed
2. **Dependencies** installed (see requirements.txt)
3. **API Key** for Polygon.io (for market data)
4. **Basic understanding** of the existing real-time system

## Installation

### 1. Install Dependencies

```bash
cd realtime/
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the realtime directory:

```bash
# Polygon.io API Key
POLYGON_API_KEY=your_polygon_api_key_here

# Optional: Database path
DB_PATH=enhanced_realtime.db

# Optional: API configuration
API_HOST=0.0.0.0
API_PORT=5000
API_KEY=your_default_api_key
```

### 3. Create Configuration File (Optional)

Create `enhanced_config.json`:

```json
{
  "db_path": "enhanced_realtime.db",
  "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
  "update_interval": 15,
  "api": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false,
    "require_auth": true,
    "api_keys": {
      "default": "default_api_key",
      "premium": "premium_api_key",
      "admin": "admin_api_key"
    },
    "rate_limits": {
      "default": {"window": 60, "requests": 100},
      "premium": {"window": 60, "requests": 1000},
      "admin": {"window": 60, "requests": 10000}
    }
  },
  "logging": {
    "log_directory": "logs",
    "max_log_file_size": 10485760,
    "log_backup_count": 5,
    "error_rate_threshold": 10
  },
  "webhooks": {
    "delivery_workers": 5,
    "delivery_queue_size": 1000,
    "default_timeout": 30
  }
}
```

## Basic Usage

### 1. Start the Enhanced System

```python
from realtime.core.enhanced_system import create_enhanced_system

# Create system with configuration
system = create_enhanced_system("enhanced_config.json")

# Start the system
system.start()

print("Enhanced system started successfully!")
```

### 2. Check System Status

```python
# Get comprehensive system status
status = system.get_system_status()
print(f"System running: {status['system_status']['started']}")
print(f"Uptime: {status['uptime_seconds']} seconds")

# Get component status
components = status['system_status']['components']
for component, healthy in components.items():
    print(f"{component}: {'✓' if healthy else '✗'}")
```

### 3. Access the API

The API server will be running on `http://localhost:5000` (or your configured host/port).

#### Test API Health

```bash
curl http://localhost:5000/health
```

#### Get Predictions (with authentication)

```bash
curl -H "X-API-Key: default_api_key" http://localhost:5000/predictions
```

#### Get Historical Performance

```bash
curl -H "X-API-Key: default_api_key" \
     "http://localhost:5000/historical/performance?symbol=AAPL&days_back=30"
```

## Key Features Demo

### 1. Real-time Data Streaming

Connect to the prediction stream:

```javascript
// WebSocket connection for real-time predictions
const eventSource = new EventSource('http://localhost:5000/stream/predictions?api_key=default_api_key');

eventSource.onmessage = function(event) {
    const predictions = JSON.parse(event.data);
    console.log('New predictions:', predictions);
};
```

### 2. Data Export

Export predictions to CSV:

```python
# Export recent predictions
export_id = system.export_data(
    export_type="predictions",
    format="csv",
    symbols=["AAPL", "GOOGL"],
    start_date="2024-01-01",
    end_date="2024-01-31"
)

# Check export status
status = system.get_export_status(export_id)
print(f"Export status: {status['status']}")
```

### 3. Webhook Notifications

Set up webhook for prediction updates:

```python
# Create webhook
webhook_id = system.create_webhook(
    name="My Prediction Webhook",
    url="https://your-server.com/webhook/predictions",
    event_types=["prediction_update", "trade_recommendation"],
    secret_key="your_secret_key"
)

print(f"Webhook created: {webhook_id}")
```

### 4. Error Tracking

Monitor and resolve errors:

```python
# Get recent error analysis
error_analysis = system.error_tracker.get_error_analysis(hours_back=24)
print(f"Total errors: {error_analysis['total_errors']}")
print(f"Critical errors: {len(error_analysis['critical_errors'])}")

# Resolve an error
if error_analysis['critical_errors']:
    error_id = error_analysis['critical_errors'][0].error_id
    success = system.resolve_error(
        error_id=error_id,
        resolution_notes="Fixed database connection issue",
        assigned_to="admin"
    )
    print(f"Error resolved: {success}")
```

### 5. Performance Monitoring

Monitor system performance:

```python
# Get performance summary
perf_summary = system.performance_profiler.get_performance_summary(hours_back=24)
print(f"Total profiles: {perf_summary['total_profiles']}")
print(f"Function profiles: {len(perf_summary['function_profiles'])}")

# Get system metrics
metrics = system.get_system_metrics(hours_back=24)
print(f"Error rate: {metrics['error_analysis']['error_rate']}")
```

### 6. Debug Mode

Enable debug modes for development:

```python
# Enable detailed timing
system.set_debug_mode("detailed_timing", True)

# Enable SQL query logging
system.set_debug_mode("sql_queries", True)

# Enable API request logging
system.set_debug_mode("api_requests", True)

# Check if debug mode is enabled
if system.logging_system.is_debug_mode("detailed_timing"):
    print("Detailed timing debug mode is enabled")
```

## Advanced Usage

### 1. Custom API Key Tiers

```python
# Create system with custom API keys
config = {
    "api": {
        "api_keys": {
            "basic": "basic_key_123",
            "premium": "premium_key_456",
            "enterprise": "enterprise_key_789"
        },
        "rate_limits": {
            "basic": {"window": 60, "requests": 50},
            "premium": {"window": 60, "requests": 500},
            "enterprise": {"window": 60, "requests": 5000}
        }
    }
}

system = create_enhanced_system()
```

### 2. Webhook Filtering

```python
# Create webhook with filters
webhook_id = system.create_webhook(
    name="AAPL Only Webhook",
    url="https://your-server.com/webhook/aapl",
    event_types=["prediction_update"],
    filters={
        "symbol": "AAPL",
        "confidence_score": {"min": 0.8}
    }
)
```

### 3. Performance Profiling Decorators

```python
from realtime.core.performance_profiler import profile_function_calls
from realtime.core.error_tracker import track_function_errors

@profile_function_calls(system.performance_profiler)
@track_function_errors(system.error_tracker)
def my_prediction_function(symbol):
    # This function will be automatically profiled and error-tracked
    return make_prediction(symbol)
```

### 4. Log Search and Analysis

```python
# Search logs with filters
logs = system.search_logs({
    "level": "ERROR",
    "category": "prediction",
    "symbol": "AAPL"
}, limit=50)

for log in logs:
    print(f"{log['timestamp']}: {log['message']}")

# Get log statistics
log_stats = system.logging_system.get_log_statistics()
print(f"Total logs: {log_stats['processor_stats']['total_logs']}")
```

## Monitoring and Maintenance

### 1. Health Monitoring

```python
# The system automatically monitors health, but you can check manually
status = system.get_system_status()

# Check for critical issues
if 'critical_errors' in status:
    print(f"Critical errors found: {len(status['critical_errors'])}")
```

### 2. Log Cleanup

```python
# Clean up old logs (older than 30 days)
success = system.logging_system.cleanup_old_logs(days_back=30)
print(f"Log cleanup: {'Success' if success else 'Failed'}")
```

### 3. Performance Optimization

```python
# Get performance recommendations
perf_summary = system.performance_profiler.get_performance_summary()
print("Performance summary:", perf_summary)

# Check for performance alerts
if 'performance_alerts' in perf_summary:
    for alert in perf_summary['performance_alerts']:
        print(f"Performance alert: {alert}")
```

## Troubleshooting

### Common Issues

1. **API Authentication Errors**
   - Check API key configuration
   - Verify rate limits
   - Ensure proper headers

2. **Webhook Delivery Failures**
   - Check webhook URL accessibility
   - Verify signature validation
   - Check retry configuration

3. **High Error Rates**
   - Review error logs
   - Check system resources
   - Verify data source connectivity

4. **Performance Issues**
   - Monitor system resources
   - Check database performance
   - Review profiling data

### Debug Commands

```python
# Enable all debug modes
for mode in ["detailed_timing", "sql_queries", "api_requests", "model_predictions"]:
    system.set_debug_mode(mode, True)

# Get detailed system metrics
metrics = system.get_system_metrics(hours_back=1)
print("System metrics:", json.dumps(metrics, indent=2))
```

## Next Steps

1. **Explore the API**: Use the API documentation to explore all available endpoints
2. **Set up Webhooks**: Configure webhooks for your specific use cases
3. **Monitor Performance**: Use the built-in monitoring tools to optimize performance
4. **Customize Configuration**: Adjust settings based on your requirements
5. **Integrate with External Systems**: Use the API and webhooks to integrate with other systems

## Support

For additional help:
- Check the comprehensive documentation in `docs/ENHANCED_FEATURES.md`
- Review the API endpoints and examples
- Use the built-in health monitoring and error tracking
- Enable debug modes for detailed troubleshooting

The enhanced system provides a robust foundation for real-time LSTM prediction with comprehensive monitoring, logging, and integration capabilities.
