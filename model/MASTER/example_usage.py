"""
Example Usage of Enhanced MASTER Model with Polygon.io Integration
Demonstrates the complete workflow from data fetching to model training
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Basic example of using the enhanced MASTER model"""
    print("=" * 60)
    print("ENHANCED MASTER MODEL - BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    try:
        from train_enhanced_master import EnhancedMASTERTrainer
        
        # Configuration for enhanced MASTER
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
            'n_epochs': 5,                    # Number of epochs (reduced for demo)
            'lr': 1e-5,                       # Learning rate
            'batch_size': 16,                 # Batch size (reduced for demo)
            'patience': 3,                    # Early stopping patience
            
            # Data
            'lookback_window': 8,             # Time window
            'seed': 42                        # Random seed
        }
        
        # Example tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        from_date = '2024-01-01'
        to_date = '2024-01-31'
        
        print(f"Training on {len(tickers)} tickers: {tickers}")
        print(f"Date range: {from_date} to {to_date}")
        print(f"Model configuration: {config['d_model']}D model, {config['num_tasks']} tasks")
        
        # Initialize trainer
        trainer = EnhancedMASTERTrainer(config)
        
        # Run training
        print("\nStarting training...")
        metrics = trainer.run_full_training(tickers, from_date, to_date)
        
        print(f"\nTraining completed!")
        print(f"Final metrics: {metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Basic usage example failed: {e}")
        return None

def example_advanced_usage():
    """Advanced example with custom feature engineering"""
    print("\n" + "=" * 60)
    print("ENHANCED MASTER MODEL - ADVANCED USAGE EXAMPLE")
    print("=" * 60)
    
    try:
        from feature_engineering_pipeline import MASTERFeaturePipeline
        from enhanced_master import EnhancedMASTERModel
        from polygon_data_fetcher import PolygonDataFetcher
        from news_sentiment_processor import NewsSentimentProcessor
        from trading_schemes_features import TradingSchemesFeatures
        
        # Initialize components
        pipeline = MASTERFeaturePipeline()
        fetcher = PolygonDataFetcher()
        sentiment_processor = NewsSentimentProcessor()
        trading_schemes = TradingSchemesFeatures()
        
        print("Components initialized successfully!")
        
        # Example: Fetch data for a single ticker
        ticker = 'AAPL'
        from_date = '2024-01-01'
        to_date = '2024-01-31'
        
        print(f"\nFetching data for {ticker}...")
        
        # Get market data
        market_data = fetcher.get_technical_indicators(ticker, from_date, to_date)
        print(f"Market data shape: {market_data.shape}")
        
        # Get news sentiment
        news_data = fetcher.get_news_sentiment(ticker, from_date, to_date, limit=10)
        if not news_data.empty:
            processed_news = sentiment_processor.process_news_batch(news_data)
            print(f"News data shape: {processed_news.shape}")
        else:
            print("No news data found")
        
        # Calculate trading schemes
        if not market_data.empty:
            trading_features = trading_schemes.calculate_all_trading_schemes(market_data)
            print(f"Trading schemes shape: {trading_features.shape}")
            
            # Show some features
            signal_cols = [col for col in trading_features.columns if 'signal_weighted' in col]
            print(f"Trading scheme signals: {signal_cols}")
        
        # Example: Create custom model
        model = EnhancedMASTERModel(
            d_feat=200,
            d_model=256,
            t_nhead=4,
            s_nhead=2,
            gate_input_start_index=200,
            gate_input_end_index=263,
            T_dropout_rate=0.5,
            S_dropout_rate=0.5,
            beta=5,
            num_tasks=8,
            task_weights=[1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            n_epochs=5,
            lr=1e-5,
            GPU=0,
            train_stop_loss_thred=0.95,
            seed=42
        )
        
        print(f"\nCustom model created with {model.num_tasks} tasks")
        print(f"Task weights: {model.task_weights}")
        
        return True
        
    except Exception as e:
        logger.error(f"Advanced usage example failed: {e}")
        return False

def example_data_validation():
    """Example of data validation and quality checks"""
    print("\n" + "=" * 60)
    print("DATA VALIDATION AND QUALITY CHECKS")
    print("=" * 60)
    
    try:
        from polygon_data_fetcher import PolygonDataFetcher
        
        fetcher = PolygonDataFetcher()
        
        # Example: Validate API connection
        print("Testing Polygon.io API connection...")
        try:
            # Test with a simple request
            test_data = fetcher.get_aggregates("AAPL", "2024-01-01", "2024-01-02")
            if not test_data.empty:
                print("✅ API connection successful")
                print(f"Sample data shape: {test_data.shape}")
                print(f"Columns: {list(test_data.columns)}")
            else:
                print("⚠️ API connected but no data returned")
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False
        
        # Example: Data quality checks
        if not test_data.empty:
            print("\nData quality checks:")
            print(f"- Missing values: {test_data.isnull().sum().sum()}")
            print(f"- Data types: {test_data.dtypes.value_counts().to_dict()}")
            print(f"- Date range: {test_data.index.min()} to {test_data.index.max()}")
            
            # Check for outliers
            numeric_cols = test_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in test_data.columns:
                    q1 = test_data[col].quantile(0.25)
                    q3 = test_data[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((test_data[col] < q1 - 1.5 * iqr) | (test_data[col] > q3 + 1.5 * iqr)).sum()
                    print(f"- {col} outliers: {outliers}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data validation example failed: {e}")
        return False

def example_performance_analysis():
    """Example of performance analysis and metrics"""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS AND METRICS")
    print("=" * 60)
    
    try:
        # Simulate some performance metrics
        metrics = {
            'IC': 0.15,
            'ICIR': 0.8,
            'RIC': 0.12,
            'RICIR': 0.7,
            'MSE': 0.02,
            'MAE': 0.1
        }
        
        print("Example performance metrics:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.4f}")
        
        # Performance interpretation
        print("\nPerformance interpretation:")
        if metrics['IC'] > 0.1:
            print("✅ Good IC - model shows predictive power")
        else:
            print("⚠️ Low IC - model may need improvement")
        
        if metrics['ICIR'] > 0.5:
            print("✅ Good ICIR - stable predictions")
        else:
            print("⚠️ Low ICIR - predictions may be unstable")
        
        if metrics['MSE'] < 0.05:
            print("✅ Low MSE - good regression accuracy")
        else:
            print("⚠️ High MSE - regression accuracy needs improvement")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Performance analysis example failed: {e}")
        return None

def main():
    """Main function to run all examples"""
    print("ENHANCED MASTER MODEL - COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️ Warning: .env file not found. Please create one with your Polygon.io API key:")
        print("echo 'POLYGON_API_KEY=your_api_key_here' > .env")
        print("\nContinuing with examples that don't require API access...")
    
    # Run examples
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Advanced Usage", example_advanced_usage),
        ("Data Validation", example_data_validation),
        ("Performance Analysis", example_performance_analysis)
    ]
    
    results = {}
    
    for name, func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = func()
            results[name] = result
            if result is not None:
                print(f"✅ {name} completed successfully")
            else:
                print(f"⚠️ {name} completed with warnings")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    print(f"Examples completed: {successful}/{total}")
    
    if successful == total:
        print("🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Set up your Polygon.io API key in .env file")
        print("2. Install dependencies: pip install -r requirements_enhanced.txt")
        print("3. Run the training script: python train_enhanced_master.py")
    else:
        print("⚠️ Some examples failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check your Polygon.io API key")
        print("3. Verify internet connection for model downloads")

if __name__ == "__main__":
    main()
