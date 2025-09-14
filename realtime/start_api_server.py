#!/usr/bin/env python3
"""
API Server Startup Script
Starts the API server with a proper RealTimeSystem instance
"""

import logging
import sys
import os
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.api_server import APIServer
from core.realtime_system import RealTimeSystem
from core.config import Config

def main():
    """Start the API server with RealTimeSystem"""
    print("🚀 Starting API Server with RealTimeSystem...")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_file = os.getenv('CONFIG_FILE', 'realtime_config.json')
        if os.path.exists(config_file):
            config = Config(config_file)
            system_config = config.get_system_config()
        else:
            # Use default configuration
            system_config = {
                'db_path': 'realtime.db',
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'update_interval': 15,
                'min_profit_percent': 2.0,
                'confidence_threshold': 0.7,
                'models_dir': 'lstms',
                'api': {
                    'host': '0.0.0.0',
                    'port': 5000,
                    'debug': False,
                    'require_auth': False,  # Disable auth for testing
                    'api_keys': {'default': 'test_key'},
                    'rate_limits': {
                        'default': {'window': 60, 'requests': 1000}
                    }
                }
            }
        
        # Create RealTimeSystem instance
        logger.info("Creating RealTimeSystem instance...")
        realtime_system = RealTimeSystem(system_config)
        
        # Start the RealTimeSystem in a separate thread to avoid blocking
        logger.info("Starting RealTimeSystem in background thread...")
        import threading
        realtime_thread = threading.Thread(target=realtime_system.start, daemon=True)
        realtime_thread.start()
        
        # Wait a moment for the system to initialize
        time.sleep(2)
        
        # Create API server
        logger.info("Creating API server...")
        api_server = APIServer(realtime_system, system_config)
        
        # Start the API server
        logger.info(f"Starting API server on {api_server.host}:{api_server.port}")
        print(f"✅ API Server is running on http://localhost:{api_server.port}")
        print("📋 Available endpoints:")
        print("   - GET  /health - Health check")
        print("   - GET  /status - System status")
        print("   - GET  /predictions - Get all predictions")
        print("   - GET  /predictions/<symbol> - Get predictions for symbol")
        print("   - GET  /recommendations - Get trade recommendations")
        print("   - GET  /performance - Get performance metrics")
        print("   - POST /control/update - Force update")
        print("\n🛑 Press Ctrl+C to stop the server")
        
        api_server.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down API server...")
        if 'api_server' in locals():
            api_server.stop()
        if 'realtime_system' in locals():
            realtime_system.stop()
        print("✅ API server stopped")
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
