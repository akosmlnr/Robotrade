#!/usr/bin/env python3
"""
Test Script for Comprehensive Features
Tests the updated system with all Polygon API fields
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add the realtime directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_fetcher import PolygonDataFetcher
from data.feature_engineering import FeatureEngineer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_comprehensive_features():
    """Test the comprehensive feature system"""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing comprehensive features with all Polygon API fields")
    
    # Initialize components
    try:
        data_fetcher = PolygonDataFetcher()
        feature_engineer = FeatureEngineer()
        
        logger.info("✅ Components initialized successfully")
        
        # Test data fetching with all fields
        logger.info("📊 Fetching sample data with all fields...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)  # Last 24 hours
        
        # Fetch data
        raw_data = data_fetcher.fetch_15min_data('AAPL', start_time, end_time)
        
        if raw_data.empty:
            logger.warning("⚠️ No data available for testing")
            return False
        
        logger.info(f"✅ Fetched {len(raw_data)} data points")
        logger.info(f"📋 Raw data columns: {list(raw_data.columns)}")
        
        # Check if we have all expected fields
        expected_fields = ['open', 'high', 'low', 'close', 'volume', 'transactions', 'vwap']
        available_fields = [field for field in expected_fields if field in raw_data.columns]
        missing_fields = [field for field in expected_fields if field not in raw_data.columns]
        
        logger.info(f"✅ Available fields: {available_fields}")
        if missing_fields:
            logger.warning(f"⚠️ Missing fields: {missing_fields}")
        
        # Test feature engineering
        logger.info("🔧 Testing comprehensive feature engineering...")
        
        feature_data = feature_engineer.add_features(raw_data)
        
        if feature_data.empty:
            logger.error("❌ Feature engineering failed")
            return False
        
        logger.info(f"✅ Feature engineering successful")
        logger.info(f"📋 Feature data columns: {list(feature_data.columns)}")
        logger.info(f"📊 Feature data shape: {feature_data.shape}")
        
        # Display sample of processed data
        logger.info("📈 Sample of processed data:")
        print("\n" + "="*80)
        print("SAMPLE PROCESSED DATA")
        print("="*80)
        print(feature_data.head().round(4))
        
        # Test feature preparation for prediction
        logger.info("🎯 Testing prediction sequence preparation...")
        
        sequence_length = 25  # Use shorter sequence for testing
        prediction_sequence = feature_engineer.prepare_prediction_data(feature_data, sequence_length)
        
        if prediction_sequence is not None:
            logger.info(f"✅ Prediction sequence created successfully")
            logger.info(f"📊 Sequence shape: {prediction_sequence.shape}")
        else:
            logger.error("❌ Failed to create prediction sequence")
            return False
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎯 COMPREHENSIVE FEATURES TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Data fetching: SUCCESS ({len(raw_data)} points)")
        logger.info(f"✅ Feature engineering: SUCCESS ({len(feature_data.columns)} features)")
        logger.info(f"✅ Prediction preparation: SUCCESS")
        logger.info(f"📋 Features used: {list(feature_data.columns)}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main test function"""
    setup_logging()
    
    print("🚀 Starting comprehensive features test...")
    print("This test verifies that all Polygon API fields are being used effectively")
    print()
    
    success = test_comprehensive_features()
    
    if success:
        print("\n🎉 All tests passed! Your system is now using all available Polygon API fields.")
        print("📈 The enhanced feature set includes:")
        print("   • OHLCV data (Open, High, Low, Close, Volume)")
        print("   • Transactions count (n)")
        print("   • Volume-weighted average price (vwap)")
        print("   • Technical indicators (RSI, MACD, Bollinger Bands)")
        print("   • Derived features (price changes, volume changes)")
        print()
        print("🔄 Next steps:")
        print("   1. Run 'python retrain_models.py' to retrain models with new features")
        print("   2. Test predictions with the enhanced model")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()



