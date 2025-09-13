"""
Configuration management for Real-time LSTM Prediction System
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration manager for the real-time prediction system
    """
    
    def __init__(self, config_file: str = "realtime_config.json"):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            else:
                logger.info("No config file found, using default configuration")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # System settings
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
            "update_interval": 15,  # minutes
            "min_profit_percent": 2.0,
            "validation_threshold": 0.02,  # 2% difference threshold for reprediction
            "confidence_threshold": 0.6,  # Minimum confidence for trade recommendations
            "max_recommendations_per_symbol": 5,
            
            # API settings
            "polygon_api_key": None,  # Will use environment variable POLYGON_API_KEY
            "rate_limit": 100,  # API calls per minute
            
            # Database settings
            "db_path": "realtime_data.db",
            "data_retention_days": 30,
            
            # Model settings
            "models_dir": "models",
            "model_cache_max_age_hours": 24,
            
            # Logging settings
            "log_level": "INFO",
            "log_file": "realtime_system.log",
            
            # Performance settings
            "prediction_cache_size": 100,
            "max_concurrent_symbols": 10,
            
            # Trading settings
            "min_trade_duration_hours": 1,
            "max_trade_duration_hours": 168,  # 1 week
            "stop_loss_percent": 5.0,
            "take_profit_percent": 10.0
        }
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self.config.update(updates)
    
    def get_symbols(self) -> list:
        """Get list of configured symbols"""
        return self.config.get('symbols', [])
    
    def add_symbol(self, symbol: str) -> bool:
        """Add a symbol to the configuration"""
        symbols = self.get_symbols()
        if symbol not in symbols:
            symbols.append(symbol)
            self.set('symbols', symbols)
            return True
        return False
    
    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from the configuration"""
        symbols = self.get_symbols()
        if symbol in symbols:
            symbols.remove(symbol)
            self.set('symbols', symbols)
            return True
        return False
    
    def get_api_key(self) -> Optional[str]:
        """Get Polygon.io API key from config or environment"""
        api_key = self.get('polygon_api_key')
        if api_key:
            return api_key
        
        # Try environment variable
        return os.getenv('POLYGON_API_KEY')
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration and return validation results
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required settings
        required_settings = ['symbols', 'update_interval', 'min_profit_percent']
        for setting in required_settings:
            if setting not in self.config:
                validation_results['errors'].append(f"Missing required setting: {setting}")
                validation_results['valid'] = False
        
        # Check symbols
        symbols = self.get_symbols()
        if not symbols:
            validation_results['errors'].append("No symbols configured")
            validation_results['valid'] = False
        elif len(symbols) > 20:
            validation_results['warnings'].append("Too many symbols configured (max recommended: 20)")
        
        # Check API key
        api_key = self.get_api_key()
        if not api_key:
            validation_results['errors'].append("Polygon.io API key not configured")
            validation_results['valid'] = False
        
        # Check update interval
        update_interval = self.get('update_interval')
        if update_interval < 1 or update_interval > 60:
            validation_results['warnings'].append("Update interval should be between 1 and 60 minutes")
        
        # Check profit threshold
        min_profit = self.get('min_profit_percent')
        if min_profit < 0.1 or min_profit > 50:
            validation_results['warnings'].append("Min profit percent should be between 0.1% and 50%")
        
        return validation_results
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        return {
            'min_profit_percent': self.get('min_profit_percent'),
            'confidence_threshold': self.get('confidence_threshold'),
            'max_recommendations_per_symbol': self.get('max_recommendations_per_symbol'),
            'min_trade_duration_hours': self.get('min_trade_duration_hours'),
            'max_trade_duration_hours': self.get('max_trade_duration_hours'),
            'stop_loss_percent': self.get('stop_loss_percent'),
            'take_profit_percent': self.get('take_profit_percent')
        }
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system-specific configuration"""
        return {
            'symbols': self.get_symbols(),
            'update_interval': self.get('update_interval'),
            'validation_threshold': self.get('validation_threshold'),
            'polygon_api_key': self.get_api_key(),
            'rate_limit': self.get('rate_limit'),
            'db_path': self.get('db_path'),
            'models_dir': self.get('models_dir'),
            'log_level': self.get('log_level'),
            'log_file': self.get('log_file')
        }


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = Config("test_config.json")
    
    # Test basic operations
    print(f"Current symbols: {config.get_symbols()}")
    print(f"Update interval: {config.get('update_interval')} minutes")
    print(f"Min profit percent: {config.get('min_profit_percent')}%")
    
    # Test validation
    validation = config.validate_config()
    print(f"Config validation: {validation}")
    
    # Test adding/removing symbols
    config.add_symbol("NVDA")
    print(f"After adding NVDA: {config.get_symbols()}")
    
    config.remove_symbol("NVDA")
    print(f"After removing NVDA: {config.get_symbols()}")
    
    # Save configuration
    config.save_config()
    
    # Cleanup
    import os
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
    
    print("Configuration test completed")
