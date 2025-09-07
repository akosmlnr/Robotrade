# stock_prediction_realtime.py

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import argparse
from threading import Thread
import json

class RealTimeStockPredictor:
    def __init__(self, ticker, model_path, time_steps=3, update_interval=300):
        """
        Initialize the real-time stock predictor
        
        Args:
            ticker: Stock ticker symbol (e.g., 'GOOG', 'AAPL')
            model_path: Path to the trained model weights (.h5 file)
            time_steps: Number of time steps for LSTM (default: 3)
            update_interval: Update interval in seconds (default: 300 = 5 minutes)
        """
        self.ticker = ticker
        self.model_path = model_path
        self.time_steps = time_steps
        self.update_interval = update_interval
        self.model = None
        self.min_max_scaler = None
        self.is_running = False
        self.predictions_history = []
        
        # Load the trained model
        self.load_model()
        
    def load_model(self):
        """Load the trained LSTM model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"Model summary:")
            self.model.summary()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def get_latest_data(self, days_back=30):
        """
        Get the latest stock data from Yahoo Finance
        
        Args:
            days_back: Number of days to fetch (default: 30)
        
        Returns:
            pandas.DataFrame: Latest stock data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            print(f"📊 Fetching data for {self.ticker} from {start_date.date()} to {end_date.date()}")
            
            data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Use only Close prices
            data = data[['Close']].dropna()
            print(f"✅ Retrieved {len(data)} data points")
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise
    
    def prepare_scaler(self, historical_data):
        """
        Prepare the MinMaxScaler using historical data
        
        Args:
            historical_data: Historical stock data for fitting the scaler
        """
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.min_max_scaler.fit(historical_data)
        print("✅ Scaler fitted with historical data")
    
    def prepare_prediction_data(self, data):
        """
        Prepare data for prediction using the same format as training
        
        Args:
            data: Latest stock data
            
        Returns:
            numpy.array: Prepared data for prediction
        """
        if self.min_max_scaler is None:
            raise ValueError("Scaler not initialized. Call prepare_scaler() first.")
        
        # Scale the data
        scaled_data = self.min_max_scaler.transform(data)
        
        # Create time steps
        if len(scaled_data) < self.time_steps:
            raise ValueError(f"Not enough data points. Need at least {self.time_steps}, got {len(scaled_data)}")
        
        # Take the last time_steps for prediction
        prediction_input = scaled_data[-self.time_steps:].reshape(1, self.time_steps, 1)
        
        return prediction_input
    
    def make_prediction(self, data):
        """
        Make a prediction using the loaded model
        
        Args:
            data: Prepared prediction data
            
        Returns:
            float: Predicted stock price
        """
        try:
            prediction_scaled = self.model.predict(data, verbose=0)
            prediction = self.min_max_scaler.inverse_transform(prediction_scaled)[0][0]
            return prediction
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            raise
    
    def get_current_price(self):
        """Get the current stock price"""
        try:
            ticker = yf.Ticker(self.ticker)
            info = ticker.info
            current_price = info.get('regularMarketPrice', info.get('currentPrice'))
            return current_price
        except Exception as e:
            print(f"❌ Error getting current price: {e}")
            return None
    
    def predict_next_price(self):
        """
        Get latest data and make a prediction for the next price
        
        Returns:
            dict: Prediction results
        """
        try:
            # Get latest data
            latest_data = self.get_latest_data()
            
            # Prepare scaler if not already done
            if self.min_max_scaler is None:
                # Use more historical data for better scaler fitting
                historical_data = self.get_latest_data(days_back=365)
                self.prepare_scaler(historical_data)
            
            # Prepare prediction data
            prediction_input = self.prepare_prediction_data(latest_data)
            
            # Make prediction
            predicted_price = self.make_prediction(prediction_input)
            
            # Get current price for comparison
            current_price = self.get_current_price()
            
            # Calculate change
            price_change = None
            price_change_percent = None
            if current_price:
                price_change = predicted_price - current_price
                price_change_percent = (price_change / current_price) * 100
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'ticker': self.ticker,
                'current_price': current_price,
                'predicted_price': round(predicted_price, 2),
                'price_change': round(price_change, 2) if price_change else None,
                'price_change_percent': round(price_change_percent, 2) if price_change_percent else None,
                'last_close': round(latest_data['Close'].iloc[-1], 2)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return None
    
    def start_realtime_prediction(self):
        """Start real-time prediction loop"""
        print(f"�� Starting real-time prediction for {self.ticker}")
        print(f"⏰ Update interval: {self.update_interval} seconds")
        print("Press Ctrl+C to stop")
        
        self.is_running = True
        
        try:
            while self.is_running:
                print(f"\n{'='*50}")
                print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                result = self.predict_next_price()
                
                if result:
                    print(f"�� {result['ticker']} Prediction Results:")
                    print(f"   Current Price: ${result['current_price']}")
                    print(f"   Predicted Price: ${result['predicted_price']}")
                    print(f"   Last Close: ${result['last_close']}")
                    
                    if result['price_change']:
                        direction = "📈" if result['price_change'] > 0 else "📉"
                        print(f"   Expected Change: {direction} ${result['price_change']} ({result['price_change_percent']}%)")
                    
                    # Store prediction history
                    self.predictions_history.append(result)
                    
                    # Keep only last 100 predictions
                    if len(self.predictions_history) > 100:
                        self.predictions_history = self.predictions_history[-100:]
                
                print(f"⏳ Waiting {self.update_interval} seconds for next update...")
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n�� Stopping real-time prediction...")
            self.is_running = False
    
    def save_predictions_history(self, filename=None):
        """Save prediction history to JSON file"""
        if not filename:
            filename = f"{self.ticker}_predictions_history.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.predictions_history, f, indent=2)
            print(f"✅ Prediction history saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving history: {e}")
    
    def plot_prediction_trend(self):
        """Plot the prediction trend over time"""
        if len(self.predictions_history) < 2:
            print("❌ Not enough prediction data to plot")
            return
        
        try:
            df = pd.DataFrame(self.predictions_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['current_price'], label='Current Price', marker='o')
            plt.plot(df['timestamp'], df['predicted_price'], label='Predicted Price', marker='s')
            
            plt.title(f'{self.ticker} Real-Time Predictions')
            plt.xlabel('Time')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            filename = f"{self.ticker}_realtime_predictions.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Plot saved as {filename}")
            
        except Exception as e:
            print(f"❌ Error plotting: {e}")


def main():
    parser = argparse.ArgumentParser(description="Real-time stock prediction using LSTM")
    parser.add_argument("-ticker", required=True, help="Stock ticker symbol (e.g., GOOG, AAPL)")
    parser.add_argument("-model", required=True, help="Path to trained model (.h5 file)")
    parser.add_argument("-time_steps", type=int, default=3, help="Number of time steps (default: 3)")
    parser.add_argument("-interval", type=int, default=300, help="Update interval in seconds (default: 300)")
    parser.add_argument("-single", action="store_true", help="Make single prediction instead of continuous")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RealTimeStockPredictor(
        ticker=args.ticker,
        model_path=args.model,
        time_steps=args.time_steps,
        update_interval=args.interval
    )
    
    if args.single:
        # Make single prediction
        print("🔮 Making single prediction...")
        result = predictor.predict_next_price()
        
        if result:
            print(f"\n�� {result['ticker']} Prediction:")
            print(f"   Current Price: ${result['current_price']}")
            print(f"   Predicted Price: ${result['predicted_price']}")
            print(f"   Last Close: ${result['last_close']}")
            
            if result['price_change']:
                direction = "📈" if result['price_change'] > 0 else "📉"
                print(f"   Expected Change: {direction} ${result['price_change']} ({result['price_change_percent']}%)")
    else:
        # Start real-time prediction
        predictor.start_realtime_prediction()
        
        # Save history when stopped
        predictor.save_predictions_history()
        predictor.plot_prediction_trend()


if __name__ == "__main__":
    main()
