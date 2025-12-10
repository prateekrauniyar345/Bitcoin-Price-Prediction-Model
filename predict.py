"""
Bitcoin Price Prediction - Command Line Interface
==================================================
Load trained models and make predictions from command line.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import argparse
import os

# Configuration
MODEL_DIR = 'notebooks/saved_model'
DATA_PATH = 'data/processed/processed_data.csv'
TIME_STEPS = 50

# Available models
MODELS = {
    'lstm_v1': 'lstm_l2_v1.keras',
    'lstm_v2': 'model_lstm_l2_v2.keras',
    'gru_v1': 'gru_l2_v1.keras'
}


def load_historical_data():
    """Load and prepare historical data."""
    df = pd.read_csv(DATA_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)
    
    # Resample to daily data
    df = df.resample('1D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return df


def get_scaler(df):
    """Fit scaler on historical data."""
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_prices)
    return scaler


def predict_price(model, scaler, input_prices):
    """
    Make a prediction using the model.
    
    Args:
        model: Loaded Keras model
        scaler: Fitted MinMaxScaler
        input_prices: List of 50 daily closing prices
    
    Returns:
        Predicted price
    """
    prices = np.array(input_prices).reshape(-1, 1)
    scaled_prices = scaler.transform(prices)
    X = scaled_prices.reshape(1, TIME_STEPS, 1)
    prediction = model.predict(X, verbose=0)
    return float(prediction[0, 0])


def main():
    parser = argparse.ArgumentParser(description='Bitcoin Price Predictor CLI')
    parser.add_argument('--model', type=str, default='lstm_v2', 
                       choices=list(MODELS.keys()),
                       help='Model to use for prediction')
    parser.add_argument('--prices', type=str, 
                       help='Comma-separated list of 50 daily closing prices')
    parser.add_argument('--use-recent', action='store_true',
                       help='Use most recent 50 days from historical data')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable Models:")
        print("-" * 40)
        for name, filename in MODELS.items():
            path = os.path.join(MODEL_DIR, filename)
            status = "[FOUND]" if os.path.exists(path) else "[NOT FOUND]"
            print(f"  {name}: {filename} {status}")
        return
    
    print("\n" + "="*60)
    print("Bitcoin Price Predictor")
    print("="*60)
    
    # Load historical data
    print("\nLoading historical data...")
    df = load_historical_data()
    scaler = get_scaler(df)
    print(f"   Data range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Load model
    model_name = args.model
    model_path = os.path.join(MODEL_DIR, MODELS[model_name])
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    print(f"\nLoading model: {model_name}...")
    model = load_model(model_path)
    print(f"   Model loaded successfully!")
    
    # Get input prices
    if args.use_recent:
        prices = df.tail(TIME_STEPS)['Close'].tolist()
        print(f"\nUsing recent {TIME_STEPS} days of data")
        print(f"   Date range: {df.tail(TIME_STEPS).index[0].date()} to {df.tail(TIME_STEPS).index[-1].date()}")
    elif args.prices:
        prices = [float(p.strip()) for p in args.prices.split(',')]
        if len(prices) != TIME_STEPS:
            print(f"[ERROR] Expected {TIME_STEPS} prices, got {len(prices)}")
            return
    else:
        print("\n[WARNING] No price data provided. Use --use-recent or --prices")
        return
    
    # Make prediction
    print(f"\nMaking prediction...")
    predicted_price = predict_price(model, scaler, prices)
    
    last_price = prices[-1]
    change = predicted_price - last_price
    change_pct = (change / last_price) * 100
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"   Model: {model_name.upper()}")
    print(f"   Last Input Price: ${last_price:,.2f}")
    print(f"   Predicted Price:  ${predicted_price:,.2f}")
    print(f"   Change:           {'+'if change >= 0 else ''}{change:,.2f} ({'+' if change_pct >= 0 else ''}{change_pct:.2f}%)")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
