# Backtester for Real-time LSTM Prediction System

This backtester simulates historical predictions by feeding a random historical week to the prediction system and then running predictions every second with incremental 15-minute data updates to imitate the real-time 15-minute update cycle.

## Features

- **Random Historical Week Selection**: Automatically selects a random historical week for backtesting
- **Incremental Data Updates**: Simulates real-time data updates by adding 15 minutes of data every second
- **Prediction Simulation**: Runs the full prediction pipeline with incremental data
- **Accuracy Metrics**: Calculates MAPE, MAE, RMSE, and confidence scores
- **Multiple Run Support**: Run multiple backtests for statistical significance
- **Results Export**: Saves detailed results in JSON format
- **Configurable Speed**: Adjust simulation speed for faster testing

## Components

### 1. HistoricalDataFetcher
- Fetches historical data for random week selection
- Creates incremental data updates for simulation
- Calculates week statistics and metrics

### 2. PredictionSimulator
- Simulates real-time predictions with incremental data
- Runs predictions every second with 15-minute data increments
- Calculates accuracy metrics against actual historical data

### 3. Backtester
- Main orchestrator that coordinates the entire simulation
- Handles multiple runs and symbol testing
- Manages results export and summary statistics

## Usage

### Command Line Interface

```bash
# Basic backtest for a single symbol
python backtester.py --symbol AAPL

# Multiple symbols with multiple runs
python backtester.py --symbols AAPL GOOGL MSFT --runs 5

# Faster simulation (2x speed)
python backtester.py --symbol AAPL --speed 2.0

# Custom date range
python backtester.py --symbol AAPL --min-date 2022-01-01 --max-date 2023-01-01

# Verbose logging
python backtester.py --symbol AAPL --verbose
```

### Programmatic Usage

```python
from backtester import Backtester
from datetime import datetime

# Create configuration
config = {
    'symbols': ['AAPL', 'GOOGL'],
    'simulation_speed': 1.0,
    'models_dir': 'lstms',
    'output_dir': 'backtest_results',
    'min_date': datetime(2022, 1, 1),
    'max_date': datetime(2023, 1, 1)
}

# Create backtester
backtester = Backtester(config)

# Run single backtest
result = backtester.run_backtest('AAPL')

# Run multiple backtests
results = backtester.run_multiple_backtests(['AAPL', 'GOOGL'], num_runs_per_symbol=3)

print(f"MAPE: {results['summary']['accuracy_metrics']['mape']['mean']:.2f}%")
```

## Configuration Options

- `symbols`: List of stock symbols to test
- `simulation_speed`: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)
- `polygon_api_key`: Polygon.io API key (uses environment variable if not provided)
- `rate_limit`: API rate limit (calls per minute)
- `models_dir`: Directory containing trained LSTM models
- `min_date`: Minimum date for historical data selection
- `max_date`: Maximum date for historical data selection
- `output_dir`: Directory to save results
- `save_results`: Whether to save results to files
- `verbose`: Enable verbose logging

## Output

The backtester generates detailed results including:

- **Prediction History**: All predictions made during the simulation
- **Accuracy Metrics**: MAPE, MAE, RMSE, and confidence scores
- **Week Statistics**: Historical week data statistics
- **Simulation Metadata**: Timing, data points, and configuration

Results are saved as JSON files in the output directory with timestamps.

## Requirements

- Python 3.7+
- Polygon.io API key
- Trained LSTM models in the specified models directory
- Required dependencies from the main realtime system

## Example Output

```json
{
  "symbol": "AAPL",
  "total_predictions": 672,
  "accuracy_metrics": {
    "overall_mape": 2.34,
    "overall_mae": 3.45,
    "overall_rmse": 4.56,
    "average_confidence": 0.78
  },
  "historical_week": {
    "start_date": "2022-06-13T00:00:00",
    "end_date": "2022-06-20T00:00:00"
  },
  "week_statistics": {
    "total_data_points": 160,
    "price_range": {
      "min": 132.50,
      "max": 138.75,
      "range_percent": 4.72
    }
  }
}
```

## Notes

- The backtester requires trained LSTM models to be available in the models directory
- Historical data is fetched from Polygon.io API (requires API key)
- Simulation runs in real-time by default but can be sped up for faster testing
- Results include both individual prediction accuracy and overall simulation metrics
- The system handles weekends and holidays by adjusting to business days
