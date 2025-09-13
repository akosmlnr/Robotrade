"""
Command Line Interface for Real-time LSTM Prediction System
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any

from .realtime_system import RealTimeSystem
from .config import Config

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def start_system(args):
    """Start the real-time prediction system"""
    print("🚀 Starting Real-time LSTM Prediction System")
    
    # Load configuration
    config = Config(args.config)
    
    # Validate configuration
    validation = config.validate_config()
    if not validation['valid']:
        print("❌ Configuration validation failed:")
        for error in validation['errors']:
            print(f"   - {error}")
        return 1
    
    if validation['warnings']:
        print("⚠️  Configuration warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
    
    # Setup logging
    setup_logging(config.get('log_level'), config.get('log_file'))
    
    # Create and start system
    system = RealTimeSystem(config.get_system_config())
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        system.stop()
        print("✅ System stopped")
    
    return 0

def show_status(args):
    """Show system status"""
    print("📊 Real-time LSTM Prediction System Status")
    
    # Load configuration
    config = Config(args.config)
    
    # Create system instance (without starting)
    system = RealTimeSystem(config.get_system_config())
    
    # Get status
    status = system.get_system_status()
    
    print(f"\n🕐 Timestamp: {status['timestamp']}")
    print(f"🔄 Running: {'Yes' if status['is_running'] else 'No'}")
    print(f"📈 Symbols: {', '.join(status['symbols'])}")
    print(f"⏱️  Update Interval: {status['update_interval']} minutes")
    
    print(f"\n🤖 Models Loaded:")
    for symbol, model_info in status['models_loaded'].items():
        status_icon = "✅" if model_info['loaded'] else "❌"
        health_icon = "🟢" if model_info['health'] == 'healthy' else "🔴"
        print(f"   {status_icon} {symbol}: {health_icon} {model_info['health']}")
    
    print(f"\n💾 Database Stats:")
    db_stats = status['database_stats']
    print(f"   📊 Market Data Records: {db_stats.get('market_data_count', 0)}")
    print(f"   🔮 Predictions: {db_stats.get('predictions_count', 0)}")
    print(f"   💼 Trade Recommendations: {db_stats.get('trade_recommendations_count', 0)}")
    print(f"   📈 Unique Symbols: {db_stats.get('unique_symbols', 0)}")
    
    print(f"\n🎯 Prediction Cache: {status['prediction_cache_size']} symbols")
    
    return 0

def show_recommendations(args):
    """Show active trade recommendations"""
    print("💼 Active Trade Recommendations")
    
    # Load configuration
    config = Config(args.config)
    
    # Create system instance
    system = RealTimeSystem(config.get_system_config())
    
    # Get recommendations
    recommendations = system.get_trade_recommendations(args.symbol)
    
    if not recommendations:
        print("📭 No active recommendations found")
        return 0
    
    print(f"\n📋 Found {len(recommendations)} active recommendations:")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['symbol']}")
        print(f"   📅 Entry: {rec['entry_time']}")
        print(f"   📅 Exit: {rec['exit_time']}")
        print(f"   💰 Entry Price: ${rec['entry_price']:.2f}")
        print(f"   💰 Exit Price: ${rec['exit_price']:.2f}")
        print(f"   📈 Expected Profit: {rec['expected_profit_percent']:.2f}%")
        print(f"   🎯 Confidence: {rec['confidence_score']:.2f}")
        print(f"   ⏱️  Duration: {rec.get('duration_hours', 'N/A')} hours")
    
    return 0

def show_predictions(args):
    """Show current predictions"""
    print("🔮 Current Predictions")
    
    # Load configuration
    config = Config(args.config)
    
    # Create system instance
    system = RealTimeSystem(config.get_system_config())
    
    # Get predictions
    predictions = system.get_current_predictions(args.symbol)
    
    if not predictions:
        print("📭 No predictions available")
        return 0
    
    if args.symbol:
        # Show predictions for specific symbol
        pred_data = predictions
        print(f"\n📈 {args.symbol} Predictions:")
        print(f"   🕐 Generated: {pred_data['timestamp']}")
        print(f"   🎯 Confidence: {pred_data['confidence_score']:.2f}")
        print(f"   📊 Total Predictions: {len(pred_data['predictions'])}")
        print(f"   💼 Recommendations: {len(pred_data['recommendations'])}")
        
        if args.show_prices:
            print(f"\n💰 Price Predictions (next 24 hours):")
            pred_df = pred_data['predictions']
            next_24h = pred_df.head(96)  # 24 hours = 96 * 15-minute intervals
            for timestamp, row in next_24h.iterrows():
                print(f"   {timestamp}: ${row['predicted_price']:.2f}")
    else:
        # Show predictions for all symbols
        print(f"\n📊 Predictions for {len(predictions)} symbols:")
        for symbol, pred_data in predictions.items():
            print(f"   📈 {symbol}: {len(pred_data['predictions'])} predictions, "
                  f"confidence {pred_data['confidence_score']:.2f}")
    
    return 0

def force_update(args):
    """Force an immediate update"""
    print("🔄 Forcing immediate update...")
    
    # Load configuration
    config = Config(args.config)
    
    # Create system instance
    system = RealTimeSystem(config.get_system_config())
    
    # Force update
    system.force_update(args.symbol)
    
    print("✅ Update completed")
    return 0

def show_config(args):
    """Show current configuration"""
    print("⚙️  Real-time LSTM Prediction System Configuration")
    
    # Load configuration
    config = Config(args.config)
    
    # Show configuration
    print(f"\n📋 Current Configuration:")
    print(f"   📈 Symbols: {', '.join(config.get_symbols())}")
    print(f"   ⏱️  Update Interval: {config.get('update_interval')} minutes")
    print(f"   💰 Min Profit Percent: {config.get('min_profit_percent')}%")
    print(f"   🎯 Confidence Threshold: {config.get('confidence_threshold')}")
    print(f"   💾 Database Path: {config.get('db_path')}")
    print(f"   🤖 Models Directory: {config.get('models_dir')}")
    print(f"   📝 Log Level: {config.get('log_level')}")
    
    # Show validation results
    validation = config.validate_config()
    print(f"\n✅ Configuration Status: {'Valid' if validation['valid'] else 'Invalid'}")
    
    if validation['errors']:
        print("❌ Errors:")
        for error in validation['errors']:
            print(f"   - {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
    
    return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Real-time LSTM Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start the system
  %(prog)s status                   # Show system status
  %(prog)s recommendations          # Show active recommendations
  %(prog)s predictions AAPL         # Show predictions for AAPL
  %(prog)s update AAPL              # Force update for AAPL
  %(prog)s config                   # Show configuration
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='realtime_config.json',
        help='Configuration file path (default: realtime_config.json)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the real-time system')
    start_parser.set_defaults(func=start_system)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.set_defaults(func=show_status)
    
    # Recommendations command
    rec_parser = subparsers.add_parser('recommendations', help='Show active trade recommendations')
    rec_parser.add_argument('symbol', nargs='?', help='Stock symbol (optional)')
    rec_parser.set_defaults(func=show_recommendations)
    
    # Predictions command
    pred_parser = subparsers.add_parser('predictions', help='Show current predictions')
    pred_parser.add_argument('symbol', nargs='?', help='Stock symbol (optional)')
    pred_parser.add_argument('--show-prices', action='store_true', help='Show detailed price predictions')
    pred_parser.set_defaults(func=show_predictions)
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Force immediate update')
    update_parser.add_argument('symbol', nargs='?', help='Stock symbol (optional)')
    update_parser.set_defaults(func=force_update)
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    config_parser.set_defaults(func=show_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        return args.func(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
