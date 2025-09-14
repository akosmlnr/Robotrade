#!/usr/bin/env python3
"""
Debug script to analyze predictions and understand why no trade recommendations are generated
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append('/app')

def analyze_predictions():
    """Analyze the predictions to understand profit opportunities"""
    
    try:
        # Import the required modules
        from storage.data_storage import DataStorage
        from models.prediction_engine import PredictionEngine
        from models.model_manager import ModelManager
        
        print("🔍 Analyzing predictions and trade recommendation criteria...")
        
        # Initialize components
        data_storage = DataStorage()
        model_manager = ModelManager()
        prediction_engine = PredictionEngine(model_manager, data_storage)
        
        symbol = 'AAPL'
        
        # Get the model
        model_data = model_manager.get_model(symbol)
        if not model_data:
            print(f"❌ No model found for {symbol}")
            return
        
        print(f"✅ Model loaded for {symbol}")
        
        # Get recent predictions
        recent_predictions = data_storage.get_recent_predictions(symbol, hours_back=24)
        if recent_predictions.empty:
            print(f"❌ No recent predictions found for {symbol}")
            return
        
        print(f"✅ Found {len(recent_predictions)} recent predictions")
        
        # Analyze profit opportunities
        predictions = recent_predictions['predicted_price'].values
        timestamps = recent_predictions.index
        
        print(f"\n📊 Prediction Analysis:")
        print(f"   First prediction: ${predictions[0]:.2f} at {timestamps[0]}")
        print(f"   Last prediction:  ${predictions[-1]:.2f} at {timestamps[-1]}")
        print(f"   Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        print(f"   Price volatility: {((predictions.max() - predictions.min()) / predictions.min() * 100):.2f}%")
        
        # Simulate the trade recommendation logic
        min_profit_percent = 2.0
        print(f"\n🎯 Trade Recommendation Criteria:")
        print(f"   Minimum profit required: {min_profit_percent}%")
        
        profitable_opportunities = 0
        max_profit_found = 0
        
        # Check for profit opportunities
        for i in range(len(predictions) - 1):
            entry_price = predictions[i]
            entry_time = timestamps[i]
            
            for j in range(i + 1, min(i + 100, len(predictions))):
                exit_price = predictions[j]
                exit_time = timestamps[j]
                
                profit_percent = ((exit_price - entry_price) / entry_price) * 100
                max_profit_found = max(max_profit_found, profit_percent)
                
                if profit_percent >= min_profit_percent:
                    profitable_opportunities += 1
                    if profitable_opportunities <= 5:  # Show first 5 opportunities
                        duration_hours = (exit_time - entry_time).total_seconds() / 3600
                        print(f"   ✅ Opportunity: {profit_percent:.2f}% profit over {duration_hours:.1f} hours")
                        print(f"      Entry: ${entry_price:.2f} at {entry_time}")
                        print(f"      Exit:  ${exit_price:.2f} at {exit_time}")
        
        print(f"\n📈 Results:")
        print(f"   Profitable opportunities found: {profitable_opportunities}")
        print(f"   Maximum profit opportunity: {max_profit_found:.2f}%")
        
        if profitable_opportunities == 0:
            print(f"\n💡 Why no trade recommendations?")
            print(f"   - No predictions show {min_profit_percent}%+ profit opportunity")
            print(f"   - Maximum profit found was only {max_profit_found:.2f}%")
            print(f"   - Consider lowering min_profit_percent from {min_profit_percent}% to {max_profit_found:.1f}%")
        
        # Check current configuration
        print(f"\n⚙️ Current Configuration:")
        print(f"   min_profit_percent: {min_profit_percent}%")
        print(f"   confidence_threshold: 0.6")
        print(f"   max_recommendations_per_symbol: 5")
        
    except Exception as e:
        print(f"❌ Error analyzing predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_predictions()
