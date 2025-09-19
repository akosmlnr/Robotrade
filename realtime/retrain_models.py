#!/usr/bin/env python3
"""
Model Retraining Script
Retrains all models with comprehensive features using all Polygon API fields
"""

import os
import sys
import logging
from typing import List

# Add the realtime directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_trainer import ModelTrainer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('model_retraining.log')
        ]
    )

def main():
    """Main retraining function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting comprehensive model retraining with all Polygon API fields")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Symbols to retrain (you can modify this list)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    logger.info(f"📈 Retraining models for symbols: {symbols}")
    
    # Training parameters
    training_params = {
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2
    }
    
    # Train all models
    results = trainer.train_multiple_symbols(symbols, **training_params)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 MODEL RETRAINING SUMMARY")
    print("="*60)
    print(f"Total symbols: {results['total_symbols']}")
    print(f"Successful trainings: {results['successful_trains']}")
    print(f"Failed trainings: {results['failed_trains']}")
    
    if results['successful_trains'] > 0:
        print("\n✅ Successfully trained models:")
        for symbol, result in results['results'].items():
            if result['success']:
                val_metrics = result['validation_metrics']
                print(f"  • {symbol}: RMSE={val_metrics['rmse']:.4f}, R²={val_metrics['r2']:.4f}")
    
    if results['failed_trains'] > 0:
        print("\n❌ Failed model trainings:")
        for symbol, result in results['results'].items():
            if not result['success']:
                print(f"  • {symbol}: {result.get('error', 'Unknown error')}")
    
    print("\n🎉 Model retraining completed!")
    print("📁 Models saved in 'lstms/' directory")
    print("📊 Check 'model_retraining.log' for detailed logs")

if __name__ == "__main__":
    main()



