"""
Test Script for Backtester
Basic tests to verify the backtester components work correctly
"""

import unittest
import os
import sys
import tempfile
import shutil
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add parent directory to path to import realtime modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from historical_data_fetcher import HistoricalDataFetcher
from prediction_simulator import PredictionSimulator
from backtester import Backtester

class TestHistoricalDataFetcher(unittest.TestCase):
    """Test the HistoricalDataFetcher class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.fetcher = HistoricalDataFetcher(api_key="test_key")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_get_random_historical_week(self):
        """Test random historical week selection"""
        symbol = "AAPL"
        start_date, end_date = self.fetcher.get_random_historical_week(symbol)
        
        # Check that dates are valid
        self.assertIsInstance(start_date, datetime)
        self.assertIsInstance(end_date, datetime)
        self.assertLess(start_date, end_date)
        
        # Check that it's approximately a week
        week_duration = (end_date - start_date).days
        self.assertGreaterEqual(week_duration, 5)  # At least 5 business days
        self.assertLessEqual(week_duration, 10)    # At most 10 days (including weekends)
    
    def test_adjust_to_business_day(self):
        """Test business day adjustment"""
        # Test weekend adjustment
        saturday = datetime(2023, 1, 7)  # Saturday
        monday = self.fetcher._adjust_to_business_day(saturday, forward=True)
        self.assertEqual(monday.weekday(), 0)  # Monday
        
        friday = self.fetcher._adjust_to_business_day(saturday, forward=False)
        self.assertEqual(friday.weekday(), 4)  # Friday
    
    def test_calculate_week_statistics(self):
        """Test week statistics calculation"""
        # Create mock data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15T')
        mock_data = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000000 + i * 1000 for i in range(100)]
        }, index=dates)
        
        stats = self.fetcher.calculate_week_statistics(mock_data)
        
        # Check that statistics are calculated
        self.assertIn('total_data_points', stats)
        self.assertIn('price_range', stats)
        self.assertIn('volatility', stats)
        self.assertIn('volume_stats', stats)
        
        self.assertEqual(stats['total_data_points'], 100)
        self.assertGreater(stats['price_range']['range'], 0)

class TestPredictionSimulator(unittest.TestCase):
    """Test the PredictionSimulator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model_manager = Mock()
        self.mock_data_storage = Mock()
        self.simulator = PredictionSimulator(
            self.mock_model_manager, 
            self.mock_data_storage,
            simulation_speed=1.0
        )
    
    def test_initialization(self):
        """Test simulator initialization"""
        self.assertFalse(self.simulator.is_running)
        self.assertIsNone(self.simulator.current_symbol)
        self.assertEqual(self.simulator.simulation_speed, 1.0)
    
    def test_get_simulation_status(self):
        """Test getting simulation status"""
        status = self.simulator.get_simulation_status()
        
        self.assertIn('is_running', status)
        self.assertIn('current_symbol', status)
        self.assertIn('total_updates', status)
        self.assertIn('current_update_index', status)
        self.assertIn('progress_percent', status)
    
    def test_stop_simulation(self):
        """Test stopping simulation"""
        # Should not raise any errors even when not running
        self.simulator.stop_simulation()
        self.assertFalse(self.simulator.is_running)

class TestBacktester(unittest.TestCase):
    """Test the Backtester class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'symbols': ['AAPL'],
            'simulation_speed': 1.0,
            'models_dir': self.temp_dir,
            'output_dir': self.temp_dir,
            'db_path': os.path.join(self.temp_dir, 'test.db')
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @patch('backtester.HistoricalDataFetcher')
    @patch('backtester.ModelManager')
    @patch('backtester.DataStorage')
    def test_initialization(self, mock_data_storage, mock_model_manager, mock_fetcher):
        """Test backtester initialization"""
        backtester = Backtester(self.config)
        
        self.assertFalse(backtester.is_running)
        self.assertIsNone(backtester.current_results)
        self.assertEqual(backtester.config['symbols'], ['AAPL'])
    
    def test_get_default_config(self):
        """Test default configuration"""
        backtester = Backtester()
        config = backtester.config
        
        self.assertIn('symbols', config)
        self.assertIn('simulation_speed', config)
        self.assertIn('models_dir', config)
        self.assertIn('output_dir', config)
        self.assertEqual(config['symbols'], ['AAPL'])
        self.assertEqual(config['simulation_speed'], 1.0)
    
    def test_get_status(self):
        """Test getting backtester status"""
        with patch('backtester.HistoricalDataFetcher'), \
             patch('backtester.ModelManager'), \
             patch('backtester.DataStorage'):
            
            backtester = Backtester(self.config)
            status = backtester.get_status()
            
            self.assertIn('is_running', status)
            self.assertIn('current_results', status)
            self.assertIn('simulation_status', status)

class TestIntegration(unittest.TestCase):
    """Integration tests for the backtester"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @patch('historical_data_fetcher.PolygonDataFetcher')
    def test_historical_fetcher_with_mock_api(self, mock_fetcher_class):
        """Test historical data fetcher with mocked API"""
        # Mock the API response
        mock_fetcher = Mock()
        mock_fetcher.fetch_15min_data.return_value = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='15T'))
        mock_fetcher_class.return_value = mock_fetcher
        
        fetcher = HistoricalDataFetcher(api_key="test_key")
        
        # Test fetching historical data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        data = fetcher.fetch_historical_week_data("AAPL", start_date, end_date)
        
        self.assertFalse(data.empty)
        self.assertEqual(len(data), 3)
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        fetcher = HistoricalDataFetcher(api_key="test_key")
        
        # Test cache info
        cache_info = fetcher.get_cache_info()
        self.assertIn('cache_size', cache_info)
        self.assertEqual(cache_info['cache_size'], 0)
        
        # Test clearing cache
        fetcher.clear_cache()
        cache_info = fetcher.get_cache_info()
        self.assertEqual(cache_info['cache_size'], 0)

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestHistoricalDataFetcher))
    test_suite.addTest(unittest.makeSuite(TestPredictionSimulator))
    test_suite.addTest(unittest.makeSuite(TestBacktester))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Running Backtester Tests")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
