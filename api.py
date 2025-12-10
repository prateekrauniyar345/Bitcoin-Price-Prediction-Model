"""
Bitcoin Price Prediction - Flask API
=====================================
A web application to predict Bitcoin prices using trained LSTM/GRU models.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Configuration
MODEL_DIR = 'notebooks/saved_model'
DATA_PATH = 'data/processed/processed_data.csv'
TIME_STEPS = 50  # Must match the training configuration

# Available models
MODELS = {
    'lstm_v1': 'lstm_l2_v1.keras',
    'lstm_v2': 'model_lstm_l2_v2.keras',
    'gru_v1': 'gru_l2_v1.keras'
}

# Global variables for model and scaler
loaded_models = {}
scaler = None
recent_data = None


def load_historical_data():
    """Load and prepare historical data for the scaler."""
    global scaler, recent_data
    
    try:
        # Load processed data
        df = pd.read_csv(DATA_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)
        
        # Resample to daily data (matching notebook preprocessing)
        df = df.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Fit scaler on all Close prices
        close_prices = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(close_prices)
        
        # Store recent data for reference
        recent_data = df.tail(TIME_STEPS + 10)
        
        print(f"[OK] Loaded historical data: {len(df)} days")
        print(f"[OK] Date range: {df.index.min()} to {df.index.max()}")
        print(f"[OK] Recent price range: ${df['Close'].min():,.2f} - ${df['Close'].max():,.2f}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        # Create a fallback scaler based on typical BTC price range
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit on approximate BTC price range (adjust based on your data)
        scaler.fit(np.array([[1000], [100000]]))
        return False


def load_all_models():
    """Load all available models."""
    global loaded_models
    
    for model_name, model_file in MODELS.items():
        model_path = os.path.join(MODEL_DIR, model_file)
        if os.path.exists(model_path):
            try:
                loaded_models[model_name] = load_model(model_path)
                print(f"[OK] Loaded model: {model_name}")
            except Exception as e:
                print(f"[ERROR] Failed to load {model_name}: {e}")
        else:
            print(f"[WARNING] Model not found: {model_path}")


def predict_price(input_prices, model_name='lstm_v2'):
    """
    Predict the next day's Bitcoin price.
    
    Args:
        input_prices: List of 50 daily closing prices
        model_name: Which model to use for prediction
    
    Returns:
        Predicted price (float)
    """
    if model_name not in loaded_models:
        raise ValueError(f"Model '{model_name}' not loaded")
    
    if len(input_prices) != TIME_STEPS:
        raise ValueError(f"Expected {TIME_STEPS} prices, got {len(input_prices)}")
    
    model = loaded_models[model_name]
    
    # Convert to numpy array and reshape
    prices = np.array(input_prices).reshape(-1, 1)
    
    # Scale the input
    scaled_prices = scaler.transform(prices)
    
    # Reshape for LSTM: (1, time_steps, 1)
    X = scaled_prices.reshape(1, TIME_STEPS, 1)
    
    # Make prediction
    prediction_scaled = model.predict(X, verbose=0)
    
    # The model outputs scaled values, but y_test was NOT scaled in the notebook
    # So the prediction is already in the original price scale
    predicted_price = float(prediction_scaled[0, 0])
    
    return predicted_price


@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html', 
                         models=list(MODELS.keys()),
                         time_steps=TIME_STEPS,
                         recent_data=recent_data.tail(TIME_STEPS)['Close'].tolist() if recent_data is not None else None)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()
        
        # Get model selection
        model_name = data.get('model', 'lstm_v2')
        
        # Get input prices
        if 'prices' in data:
            # User provided custom prices
            prices = data['prices']
            if isinstance(prices, str):
                prices = [float(p.strip()) for p in prices.split(',') if p.strip()]
        elif 'use_recent' in data and data['use_recent']:
            # Use recent historical data
            if recent_data is not None:
                prices = recent_data.tail(TIME_STEPS)['Close'].tolist()
            else:
                return jsonify({'error': 'Historical data not available'}), 400
        else:
            return jsonify({'error': 'No price data provided'}), 400
        
        # Validate input
        if len(prices) != TIME_STEPS:
            return jsonify({
                'error': f'Expected {TIME_STEPS} prices, got {len(prices)}. Please provide exactly {TIME_STEPS} daily closing prices.'
            }), 400
        
        # Make prediction
        predicted_price = predict_price(prices, model_name)
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'model_used': model_name,
            'input_prices_count': len(prices),
            'last_input_price': round(prices[-1], 2),
            'price_change': round(predicted_price - prices[-1], 2),
            'price_change_pct': round((predicted_price - prices[-1]) / prices[-1] * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recent-prices')
def get_recent_prices():
    """Get recent historical prices."""
    if recent_data is not None:
        prices = recent_data.tail(TIME_STEPS)['Close'].tolist()
        dates = recent_data.tail(TIME_STEPS).index.strftime('%Y-%m-%d').tolist()
        return jsonify({
            'prices': prices,
            'dates': dates
        })
    return jsonify({'error': 'Historical data not available'}), 400


@app.route('/api/models')
def get_models():
    """Get list of available models."""
    return jsonify({
        'available_models': list(loaded_models.keys()),
        'all_models': list(MODELS.keys())
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(loaded_models.keys()),
        'scaler_ready': scaler is not None,
        'data_loaded': recent_data is not None
    })


# Initialize on startup
print("\n" + "="*60)
print("Bitcoin Price Predictor - Initializing...")
print("="*60)

load_historical_data()
load_all_models()

print("="*60)
print(f"[READY] Models loaded: {list(loaded_models.keys())}")
print("="*60 + "\n")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
