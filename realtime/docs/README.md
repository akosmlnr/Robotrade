# Real-time LSTM Prediction System

A production-ready real-time stock prediction system that generates 1-week ahead predictions using pre-trained LSTM models and provides intelligent trade recommendations based on 15-minute delayed market data.

## 🚀 **System Overview**

- **1-Week Ahead Predictions**: Generates predictions for the next 7 days using rolling window updates
- **Real-time Data**: Uses Polygon.io API with 15-minute delayed data
- **Intelligent Validation**: Multi-factor validation with automatic reprediction
- **Advanced Confidence Scoring**: 10-factor confidence calculation system
- **Trade Recommendations**: Identifies profitable trading opportunities with entry/exit times
- **Performance Monitoring**: Comprehensive monitoring and alerting system
- **REST API**: External access and integration interface
- **Optimization**: Performance optimization and caching system
- **Enhanced Features**: Advanced logging, webhook notifications, error tracking, and performance profiling

## 📊 **Implementation Statistics**

- **Total Components**: 24+ components across 4 phases
- **Lines of Code**: ~15,000+ lines
- **API Endpoints**: 20+ RESTful endpoints
- **Features**: 100+ features and capabilities
- **Documentation**: Complete technical and user documentation

## 🚀 **Quick Start Guide**

### **1. Prerequisites**

1. **Python 3.8+** installed
2. **Dependencies** installed (see requirements.txt)
3. **API Key** for Polygon.io (for market data)
4. **Basic understanding** of the existing real-time system

### **2. Installation**

```bash
# Navigate to the realtime directory
cd realtime/

# Install dependencies
pip install -r requirements.txt
```

### **3. Environment Setup**

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

### **4. Prepare Models**

Ensure you have pre-trained LSTM models in the following structure:

```
models/
├── AAPL/
│   ├── model_weights.h5 (or model_weights.keras)
│   └── scaler.pkl
├── GOOGL/
│   ├── model_weights.h5 (or model_weights.keras)
│   └── scaler.pkl
└── ...
```

#### **Model File Requirements:**

- **Model Weights**: `model_weights.h5` or `model_weights.keras` (TensorFlow/Keras model)
- **Scaler**: `scaler.pkl` (scikit-learn MinMaxScaler or similar)
- **Configuration**: `model_config.json` (optional - will be auto-generated if missing)

#### **Auto-Configuration**

The system will automatically:
- Analyze your model architecture to determine sequence length and features
- Create a default configuration if `model_config.json` is missing
- Support both `.h5` and `.keras` model formats
- Generate appropriate feature names based on your model's input shape

### **5. Configuration**

Create a configuration file `enhanced_config.json`:

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
  },
  "confidence_factor_weights": {
    "model_performance": 0.25,
    "data_quality": 0.15,
    "market_volatility": 0.15,
    "trend_consistency": 0.10,
    "volume_analysis": 0.10,
    "technical_indicators": 0.10,
    "time_of_day": 0.05,
    "market_regime": 0.05,
    "prediction_stability": 0.03,
    "historical_accuracy": 0.02
  },
  "alert_thresholds": {
    "prediction_error_high": 0.05,
    "prediction_error_critical": 0.10,
    "confidence_low": 0.4,
    "confidence_critical": 0.2,
    "api_response_time_high": 5.0,
    "api_response_time_critical": 10.0,
    "model_accuracy_low": 0.6,
    "trade_success_rate_low": 0.4
  },
  "cache": {
    "max_size_mb": 500,
    "ttl_minutes": 60,
    "enable_prediction_cache": true,
    "enable_model_cache": true,
    "enable_data_cache": true
  }
}
```

### **6. Run the System**

#### **Enhanced System (Recommended)**

```python
from realtime.core.enhanced_system import create_enhanced_system

# Create system with configuration
system = create_enhanced_system("enhanced_config.json")

# Start the system
system.start()

print("Enhanced system started successfully!")
```

#### **CLI Interface**

```bash
# Start the real-time system
python -m realtime.cli start

# Check system status
python -m realtime.cli status

# View active recommendations
python -m realtime.cli recommendations

# View predictions for a specific symbol
python -m realtime.cli predictions AAPL

# Force an immediate update
python -m realtime.cli update AAPL
```

#### **REST API Server**

```bash
# Start the API server
python -c "
from realtime.api_server import APIServer
from realtime.realtime_system import RealTimeSystem
from realtime.config import Config

config = Config('enhanced_config.json')
system = RealTimeSystem(config.get_system_config())
api_server = APIServer(system, config.get_system_config())
api_server.start()
"
```

## 🌐 **REST API Endpoints**

### **System Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/config` | GET | System configuration |
| `/config` | POST | Update configuration |

### **Data Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predictions` | GET | All predictions |
| `/predictions/{symbol}` | GET | Symbol predictions |
| `/recommendations` | GET | All recommendations |
| `/recommendations/{symbol}` | GET | Symbol recommendations |

### **Historical Data**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/historical/performance` | GET | Historical performance data |
| `/historical/predictions` | GET | Historical predictions |

### **Data Export**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/export/csv` | POST | CSV export |
| `/export/json` | POST | JSON export |
| `/export/status/{export_id}` | GET | Export status |

### **Real-time Streaming**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream/predictions` | GET | Real-time prediction stream |
| `/stream/alerts` | GET | Real-time alert stream |

### **Webhook Management**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhooks` | GET | List webhooks |
| `/webhooks` | POST | Create webhook |
| `/webhooks/{webhook_id}` | PUT | Update webhook |
| `/webhooks/{webhook_id}` | DELETE | Delete webhook |

### **Data Visualization**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/visualization/performance` | GET | Performance charts |
| `/visualization/predictions` | GET | Prediction charts |

### **Monitoring Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/performance` | GET | System performance |
| `/performance/{symbol}` | GET | Symbol performance |
| `/alerts` | GET | Active alerts |

### **Control Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/control/update` | POST | Force update all |
| `/control/update/{symbol}` | POST | Force update symbol |

### **API Usage Examples**

```bash
# Health check
curl -X GET http://localhost:5000/health

# Get predictions (with authentication)
curl -X GET http://localhost:5000/predictions \
  -H "X-API-Key: your_api_key"

# Get symbol predictions
curl -X GET http://localhost:5000/predictions/AAPL \
  -H "X-API-Key: your_api_key"

# Get performance report
curl -X GET http://localhost:5000/performance/AAPL?period_hours=24 \
  -H "X-API-Key: your_api_key"

# Force update
curl -X POST http://localhost:5000/control/update/AAPL \
  -H "X-API-Key: your_api_key"

# Get historical performance
curl -H "X-API-Key: your_api_key" \
     "http://localhost:5000/historical/performance?symbol=AAPL&days_back=30"
```

## 🎯 **Confidence Factors**

The system uses 10 different factors to calculate prediction confidence:

1. **Model Performance** (25% weight): RMSE, R², accuracy metrics
2. **Data Quality** (15% weight): Missing values, outliers, consistency
3. **Market Volatility** (15% weight): Price volatility analysis
4. **Trend Consistency** (10% weight): Prediction vs market trend alignment
5. **Volume Analysis** (10% weight): Trading volume patterns
6. **Technical Indicators** (10% weight): Moving averages, RSI alignment
7. **Time of Day** (5% weight): Market hours vs off-hours
8. **Market Regime** (5% weight): Volatility and trend regimes
9. **Prediction Stability** (3% weight): Variance in predictions
10. **Historical Accuracy** (2% weight): Past performance tracking

### **Confidence Levels**

- **Very High** (0.85-1.0): Strong trading signal
- **High** (0.7-0.85): Good trading signal
- **Medium** (0.5-0.7): Moderate trading signal
- **Low** (0.3-0.5): Weak trading signal
- **Very Low** (0.0-0.3): Avoid trading

## 📊 **Performance Monitoring**

### **Monitoring Categories**

1. **Prediction Accuracy**: Error rates, success metrics
2. **Model Performance**: RMSE, R², accuracy trends
3. **System Health**: CPU, memory, disk usage
4. **API Performance**: Response times, error rates
5. **Database Performance**: Query times, connection health
6. **Trade Performance**: Success rates, profit metrics

### **Alert Levels**

- **INFO**: Informational messages
- **WARNING**: Performance degradation
- **ERROR**: System errors
- **CRITICAL**: System failures requiring immediate attention

### **Key Performance Indicators (KPIs)**

1. **Prediction Accuracy**: Percentage of accurate predictions
2. **Confidence Score**: Average confidence across all predictions
3. **Reprediction Rate**: Percentage of predictions requiring reprediction
4. **Cache Hit Rate**: Percentage of cache hits vs misses
5. **API Response Time**: Average API response time
6. **System Uptime**: System availability percentage
7. **Memory Usage**: Peak and average memory consumption
8. **Trade Success Rate**: Percentage of profitable trades

## 🔄 **System Workflow**

### **1. Prediction Generation Workflow**

```
1. Fetch latest market data from Polygon.io
2. Store data in SQLite database
3. Load pre-trained LSTM model for symbol
4. Generate 1-week ahead predictions (672 intervals)
5. Calculate confidence score using 10 factors
6. Validate predictions against actual data (when available)
7. Trigger reprediction if validation fails
8. Generate trade recommendations for profitable periods
9. Store predictions and recommendations in database
10. Update performance metrics and monitoring
```

### **2. Validation Workflow**

```
1. Compare predicted prices with actual market prices
2. Calculate prediction error percentage
3. Analyze market volatility and trend consistency
4. Determine if reprediction is needed
5. Trigger reprediction if error exceeds threshold
6. Update confidence scores based on validation results
7. Record performance metrics
8. Generate alerts if performance degrades
```

### **3. Optimization Workflow**

```
1. Monitor system performance metrics
2. Check cache hit rates and memory usage
3. Optimize memory usage and clean expired cache entries
4. Update performance metrics
5. Generate optimization recommendations
6. Apply performance improvements automatically
```

## 🧪 **Usage Examples**

### **Basic Usage**

```python
from realtime import RealTimeSystem, Config

# Load configuration
config = Config('enhanced_config.json')

# Create system
system = RealTimeSystem(config.get_system_config())

# Start system
system.start()

# Get current predictions
predictions = system.get_current_predictions('AAPL')
print(f"Predictions for AAPL: {predictions}")

# Get trade recommendations
recommendations = system.get_trade_recommendations('AAPL')
print(f"Recommendations: {recommendations}")

# Get system status
status = system.get_system_status()
print(f"System status: {status}")
```

### **Enhanced System Usage**

```python
from realtime.core.enhanced_system import create_enhanced_system

# Create enhanced system with configuration
system = create_enhanced_system("enhanced_config.json")

# Start the system
system.start()

# Get comprehensive system status
status = system.get_system_status()
print(f"System running: {status['system_status']['started']}")
print(f"Uptime: {status['uptime_seconds']} seconds")

# Get component status
components = status['system_status']['components']
for component, healthy in components.items():
    print(f"{component}: {'✓' if healthy else '✗'}")
```

### **Advanced Usage with Phase 2 Components**

```python
from realtime import (
    AdaptivePredictor, AdvancedConfidenceCalculator, 
    PerformanceMonitor, ValidationSystem, OptimizationSystem
)

# Initialize Phase 2 components
config = {...}  # Your configuration

# Adaptive prediction validation
predictor = AdaptivePredictor(config)
validation_result = predictor.validate_prediction(
    symbol='AAPL',
    predicted_price=150.0,
    actual_price=148.5,
    prediction_timestamp=datetime.now(),
    market_data=market_data
)

# Confidence calculation
calculator = AdvancedConfidenceCalculator(config)
confidence_result = calculator.calculate_confidence(
    symbol='AAPL',
    predictions=[150.0, 151.0, 152.0],
    market_data=market_data,
    model_performance={'rmse': 0.02, 'r2': 0.8, 'accuracy': 0.75},
    prediction_timestamp=datetime.now()
)

# Performance monitoring
monitor = PerformanceMonitor(config)
monitor.start_monitoring()
monitor.record_prediction_accuracy('AAPL', 150.0, 148.5, datetime.now())

# Optimization
optimizer = OptimizationSystem(config)
optimizer.start_optimization()
cache_key = optimizer.cache_prediction('AAPL', prediction_data)
```

### **Webhook Setup**

```python
# Create a webhook for prediction updates
webhook_id = system.create_webhook(
    name="Prediction Notifications",
    url="https://your-server.com/webhook/predictions",
    event_types=["prediction_update", "trade_recommendation"],
    secret_key="your_webhook_secret"
)

# Test the webhook
system.webhook_system.test_webhook(webhook_id)
```

### **Data Export**

```python
# Export predictions to CSV
export_id = system.export_data(
    export_type="predictions",
    format="csv",
    symbols=["AAPL", "GOOGL"],
    start_date="2024-01-01",
    end_date="2024-01-31"
)

# Check export status
status = system.get_export_status(export_id)
```

### **Error Tracking**

```python
# Track an error with context
from realtime.core.error_tracker import track_errors, ErrorCategory, ErrorSeverity

with track_errors(
    system.error_tracker,
    category=ErrorCategory.PREDICTION,
    severity=ErrorSeverity.HIGH,
    symbol="AAPL",
    operation="model_prediction"
):
    # Your code that might raise an error
    result = make_prediction("AAPL")
```

### **Performance Profiling**

```python
from realtime.core.performance_profiler import profile_function

@profile_function(system.performance_profiler)
def my_prediction_function(symbol):
    # Function execution will be automatically profiled
    return make_prediction(symbol)

# Or use context manager
with system.performance_profiler.profile_function(my_function, "prediction_operation"):
    result = my_function()
```

### **Debug Mode**

```python
# Enable detailed timing debug mode
system.set_debug_mode("detailed_timing", True)

# Enable SQL query logging
system.set_debug_mode("sql_queries", True)

# Enable API request logging
system.set_debug_mode("api_requests", True)

# Check if debug mode is enabled
if system.logging_system.is_debug_mode("detailed_timing"):
    print("Detailed timing debug mode is enabled")
```

## 🔧 **CLI Commands**

| Command | Description |
|---------|-------------|
| `start` | Start the real-time prediction system |
| `status` | Show system status and health |
| `recommendations [SYMBOL]` | Show active trade recommendations |
| `predictions [SYMBOL]` | Show current predictions |
| `update [SYMBOL]` | Force immediate update |
| `config` | Show current configuration |
| `performance` | Show performance metrics |
| `alerts` | Show active alerts |

## 🗄️ **Database Schema**

### **Core Tables**

- **market_data**: 15-minute OHLCV data
- **predictions**: Model predictions with confidence scores
- **trade_recommendations**: Generated trade signals
- **model_performance**: Model performance tracking
- **validation_results**: Prediction validation history
- **performance_metrics**: System performance tracking
- **alerts**: System alerts and notifications

### **Enhanced Tables**

- **logs**: Structured logging with metadata
- **webhooks**: Webhook configuration and delivery tracking
- **errors**: Error tracking and resolution management
- **performance_profiles**: Performance profiling data
- **exports**: Data export tracking and status

## 🚨 **Error Handling**

### **Comprehensive Error Handling**

- **API Rate Limiting**: Automatic retry with exponential backoff
- **Model Loading**: Validation and health checks
- **Data Validation**: Outlier detection and missing data handling
- **Prediction Validation**: NaN/infinite value detection
- **Database Errors**: Transaction rollback and error logging
- **Network Errors**: Connection timeout and retry logic
- **Memory Management**: Automatic cleanup and optimization

### **Alert System**

- **Real-time Monitoring**: 24/7 system health monitoring
- **Automated Alerts**: Immediate notification of issues
- **Performance Tracking**: Historical performance analysis
- **Proactive Management**: Early issue detection and resolution

## 🔒 **Security**

### **API Security**

- **Multi-tier Authentication**: Different access levels with different permissions
- **Rate Limiting**: Prevent abuse and DoS attacks
- **Input Validation**: Sanitize all API inputs
- **HTTPS**: Encrypt all API communications

### **Webhook Security**

- **HMAC Signatures**: Cryptographic signature verification
- **Retry Logic**: Protection against replay attacks
- **Timeout Configuration**: Protection against hanging requests
- **Filtering**: Event filtering to prevent information leakage

### **Data Security**

- **Encryption**: Encrypt sensitive data at rest
- **Access Control**: Role-based access control
- **Audit Logging**: Log all system access and changes
- **Backup Security**: Secure backup and recovery procedures

## 📈 **Scaling Considerations**

### **Horizontal Scaling**

- **API Load Balancing**: Multiple API server instances
- **Database Sharding**: Distribute data across multiple databases
- **Cache Distribution**: Distributed caching with Redis
- **Microservices**: Split components into separate services

### **Vertical Scaling**

- **Memory Optimization**: Efficient data structures and caching
- **CPU Optimization**: Parallel processing and batch operations
- **Storage Optimization**: Data compression and archiving
- **Network Optimization**: Connection pooling and caching

## 🧪 **Testing**

### **Testing Strategy**

- **Unit Testing**: Test each component individually
- **Integration Testing**: Test complete workflows
- **API Testing**: Test all API endpoints
- **Performance Testing**: Test system performance under load
- **End-to-End Testing**: Test complete prediction workflows

### **Test Examples**

```python
# Test prediction generation
def test_prediction_generation():
    system = RealTimeSystem(test_config)
    predictions = system.get_current_predictions('AAPL')
    assert len(predictions) > 0
    assert 'confidence_score' in predictions

# Test validation system
def test_validation():
    predictor = AdaptivePredictor(test_config)
    result = predictor.validate_prediction(
        'AAPL', 150.0, 148.5, datetime.now(), test_data
    )
    assert result.status in ['valid', 'needs_reprediction', 'invalid']

# Test API endpoints
def test_api_health():
    response = requests.get('http://localhost:5000/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **No models loaded**: Check model directory structure and file permissions
2. **API errors**: Verify Polygon.io API key and rate limits
3. **Database errors**: Check file permissions and disk space
4. **Prediction failures**: Validate model compatibility and data quality
5. **High memory usage**: Check cache sizes and TTL settings
6. **Low cache hit rates**: Review cache key generation and TTL settings
7. **High API response times**: Check database query performance
8. **Frequent repredictions**: Adjust validation thresholds

### **Debug Commands**

```bash
# Check system status
curl -X GET http://localhost:5000/status

# Get performance report
curl -X GET http://localhost:5000/performance

# Check active alerts
curl -X GET http://localhost:5000/alerts

# Force system update
curl -X POST http://localhost:5000/control/update
```

### **Logs**

Check the log files for detailed error information:
- `realtime_system.log`: Main system logs
- `api_server.log`: API server logs
- `performance_monitor.log`: Performance monitoring logs

## 📚 **Dependencies**

### **Core Dependencies**

```
tensorflow==2.15.0
keras==2.15.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
```

### **Financial Data**

```
yfinance==0.2.18
polygon-api-client==1.12.0
```

### **Real-time System**

```
schedule>=1.1.0
flask>=2.0.0
flask-cors>=3.0.0
psutil>=5.9.0
```

### **Enhanced Features**

```
requests>=2.31.0
python-dateutil>=2.8.2
tqdm>=4.65.0
```
