# Enhanced MASTER Model with Polygon.io Integration

This implementation extends the original MASTER model with comprehensive data integration from Polygon.io, including market data, news sentiment analysis, and trading scheme features.

## Features

### 🚀 **Enhanced Data Integration**
- **Polygon.io Market Data**: OHLCV, VWAP, technical indicators, market snapshots
- **News Sentiment Analysis**: FinBERT-based sentiment scoring for financial news
- **Trading Schemes**: 7 weighted trading scheme features (SMA, EMA, MACD, RSI, Bollinger Bands, Volume, VWAP)
- **Market Indices**: SPY, QQQ, IWM, and sector ETFs for broader market context

### 🧠 **Multi-Task Learning**
- **Primary Task**: Future return prediction
- **Auxiliary Tasks**: 7 trading scheme predictions
- **Weighted Loss**: Configurable task weights for balanced learning
- **Feature Gates**: Dynamic feature selection based on market conditions

### 📊 **Comprehensive Feature Engineering**
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Market Structure**: Price ranges, body sizes, shadows
- **Volatility Features**: Rolling volatility, high-low volatility
- **Momentum Features**: Price momentum, volume momentum, VWAP momentum
- **Liquidity Features**: Volume ratios, trade counts, VWAP deviation

## Architecture

```
Input: (N, T, F) where:
- N = Number of stocks
- T = Lookback window (default: 8 days)
- F = Feature dimension (market + sentiment + trading schemes)

Feature Gate → Temporal Attention → Spatial Attention → Multi-Task Head
     ↓              ↓                    ↓                    ↓
Market Context → Time Series → Cross-Stock → [Return, SMA, EMA, MACD, RSI, BB, Volume, VWAP]
```

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements_enhanced.txt
```

2. **Set up Environment Variables**:
```bash
# Create .env file
echo "POLYGON_API_KEY=your_polygon_api_key_here" > .env
```

3. **Download FinBERT Model** (first run will download automatically):
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

## Quick Start

### Basic Usage

```python
from train_enhanced_master import EnhancedMASTERTrainer

# Configuration
config = {
    'd_feat': 200,
    'd_model': 256,
    't_nhead': 4,
    's_nhead': 2,
    'dropout': 0.5,
    'gate_input_start_index': 200,
    'gate_input_end_index': 263,
    'beta': 5,
    'num_tasks': 8,
    'task_weights': [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    'n_epochs': 20,
    'lr': 1e-5,
    'batch_size': 32,
    'lookback_window': 8,
    'seed': 42
}

# Initialize trainer
trainer = EnhancedMASTERTrainer(config)

# Run training
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
metrics = trainer.run_full_training(
    tickers=tickers,
    from_date='2024-01-01',
    to_date='2024-03-31'
)
```

### Advanced Usage

```python
from feature_engineering_pipeline import MASTERFeaturePipeline
from enhanced_master import EnhancedMASTERModel

# Custom feature engineering
pipeline = MASTERFeaturePipeline()
features, labels = pipeline.run_full_pipeline(
    tickers=['AAPL', 'MSFT'],
    from_date='2024-01-01',
    to_date='2024-03-31',
    lookback_window=8
)

# Custom model training
model = EnhancedMASTERModel(
    d_feat=200,
    d_model=256,
    num_tasks=8,
    task_weights=[1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
)
```

## Data Sources

### Market Data (Polygon.io)
- **OHLCV Data**: Open, High, Low, Close, Volume
- **VWAP**: Volume-Weighted Average Price
- **Trade Counts**: Number of trades per bar
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands

### News Sentiment (Polygon.io + FinBERT)
- **News Articles**: Title, description, publisher, keywords
- **Sentiment Scores**: Positive, negative, neutral probabilities
- **Aggregated Sentiment**: Daily sentiment per stock
- **Quality Filtering**: Publisher credibility, article count thresholds

### Trading Schemes
1. **SMA Crossover**: 5-day vs 20-day Simple Moving Average
2. **EMA Crossover**: 5-day vs 20-day Exponential Moving Average
3. **MACD Histogram**: MACD line vs signal line
4. **RSI Threshold**: Distance from neutral 50 level
5. **Bollinger Bands**: Breakout signals from upper/lower bands
6. **Volume Spikes**: Volume vs rolling average
7. **VWAP Trend**: Price vs VWAP distance

## Model Configuration

### Key Parameters

```python
config = {
    # Model Architecture
    'd_feat': 200,                    # Feature dimension
    'd_model': 256,                   # Model dimension
    't_nhead': 4,                     # Temporal attention heads
    's_nhead': 2,                     # Spatial attention heads
    'dropout': 0.5,                   # Dropout rate
    
    # Feature Gates
    'gate_input_start_index': 200,    # Start of market features
    'gate_input_end_index': 263,      # End of market features
    'beta': 5,                        # Gate temperature
    
    # Multi-Task Learning
    'num_tasks': 8,                   # Number of prediction tasks
    'task_weights': [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],  # Task weights
    
    # Training
    'n_epochs': 20,                   # Number of epochs
    'lr': 1e-5,                       # Learning rate
    'batch_size': 32,                 # Batch size
    'patience': 5,                    # Early stopping patience
    
    # Data
    'lookback_window': 8,             # Time window
    'seed': 42                        # Random seed
}
```

### Task Weights Explanation

The model predicts 8 tasks with different weights:
1. **Return Prediction** (weight: 1.0) - Primary task
2. **SMA Signal** (weight: 0.2) - Auxiliary task
3. **EMA Signal** (weight: 0.2) - Auxiliary task
4. **MACD Signal** (weight: 0.2) - Auxiliary task
5. **RSI Signal** (weight: 0.2) - Auxiliary task
6. **Bollinger Bands Signal** (weight: 0.2) - Auxiliary task
7. **Volume Signal** (weight: 0.2) - Auxiliary task
8. **VWAP Signal** (weight: 0.2) - Auxiliary task

## Evaluation Metrics

### Primary Metrics
- **IC (Information Coefficient)**: Correlation between predictions and actual returns
- **ICIR (IC Information Ratio)**: IC divided by standard deviation
- **RIC (Rank IC)**: Spearman correlation
- **RICIR (Rank ICIR)**: RIC divided by standard deviation

### Additional Metrics
- **MSE (Mean Squared Error)**: Regression accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **Hit Rate**: Directional accuracy
- **Sharpe Ratio**: Risk-adjusted returns

## File Structure

```
model/MASTER/
├── polygon_data_fetcher.py          # Polygon.io data fetching
├── news_sentiment_processor.py      # FinBERT sentiment analysis
├── trading_schemes_features.py      # Trading scheme feature engineering
├── feature_engineering_pipeline.py  # Complete feature pipeline
├── enhanced_master.py               # Enhanced MASTER model
├── train_enhanced_master.py         # Training script
├── requirements_enhanced.txt         # Dependencies
└── README_enhanced.md               # This file
```

## API Usage

### Polygon.io Data Fetcher

```python
from polygon_data_fetcher import PolygonDataFetcher

fetcher = PolygonDataFetcher()

# Get market data
data = fetcher.get_technical_indicators("AAPL", "2024-01-01", "2024-01-31")

# Get news sentiment
news = fetcher.get_news_sentiment("AAPL", "2024-01-01", "2024-01-31")

# Get market indices
indices = fetcher.get_market_indices_data(["SPY", "QQQ"], "2024-01-01", "2024-01-31")
```

### News Sentiment Processor

```python
from news_sentiment_processor import NewsSentimentProcessor

processor = NewsSentimentProcessor()

# Process news batch
processed_news = processor.process_news_batch(news_df)

# Aggregate sentiment
sentiment_features = processor.aggregate_sentiment_by_time(processed_news)
```

### Trading Schemes Features

```python
from trading_schemes_features import TradingSchemesFeatures

schemes = TradingSchemesFeatures()

# Calculate all trading schemes
all_signals = schemes.calculate_all_trading_schemes(market_data)

# Create multi-task labels
labeled_data = schemes.create_trading_scheme_labels(market_data)
```

## Performance Considerations

### Memory Usage
- **Feature Dimension**: 200+ features per stock
- **Lookback Window**: 8 days (configurable)
- **Batch Size**: 32 (recommended for 8GB GPU)
- **Model Size**: ~2M parameters

### Training Time
- **Data Preparation**: 5-10 minutes (depending on ticker count)
- **Model Training**: 10-30 minutes per epoch
- **Total Training**: 1-3 hours (20 epochs)

### API Limits
- **Polygon.io**: 5 requests/minute (free tier)
- **Rate Limiting**: Built-in delays to respect limits
- **Caching**: Consider implementing for production use

## Troubleshooting

### Common Issues

1. **API Key Error**:
   ```bash
   # Check .env file
   cat .env
   # Should contain: POLYGON_API_KEY=your_key_here
   ```

2. **CUDA Out of Memory**:
   ```python
   # Reduce batch size
   config['batch_size'] = 16
   # Or reduce model size
   config['d_model'] = 128
   ```

3. **No Data Found**:
   ```python
   # Check date range
   # Ensure tickers are valid
   # Verify API key has sufficient credits
   ```

4. **FinBERT Download Issues**:
   ```python
   # Manual download
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
   model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project extends the original MASTER model with additional features and integrations. Please refer to the original MASTER paper for academic citations.

## Citation

If you use this enhanced implementation, please cite both the original MASTER paper and acknowledge the Polygon.io integration:

```bibtex
@article{master2023,
  title={MASTER: Multi-Aspect Non-local Network for Scene Text Recognition},
  author={Original MASTER Authors},
  journal={Original Journal},
  year={2023}
}
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This implementation requires a Polygon.io API key for data access. The free tier provides limited requests per minute, so consider upgrading for production use.
