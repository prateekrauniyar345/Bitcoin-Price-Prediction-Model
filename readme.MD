# Bitcoin Price Predictor

A deep learning-based Bitcoin price prediction system using LSTM and GRU neural networks. This project provides an end-to-end solution for training time-series forecasting models and serving predictions via a web interface and REST API.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Model Performance](#model-performance)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

This project implements a Bitcoin price prediction system that uses historical daily closing prices to forecast the next day's price. The system employs recurrent neural networks (LSTM and GRU) which are specifically designed to learn patterns in sequential data.

### Key Highlights

- **Multiple Models**: Three pre-trained models (LSTM v1, LSTM v2, GRU v1) with different configurations
- **Web Interface**: Modern, responsive UI for easy predictions
- **REST API**: Programmatic access for integration with other applications
- **CLI Support**: Command-line interface for batch predictions

---

## Features

| Feature | Description |
|---------|-------------|
| LSTM Models | Long Short-Term Memory networks for capturing long-term dependencies |
| GRU Model | Gated Recurrent Unit for faster inference |
| Web Dashboard | Interactive UI with model selection and visualization |
| REST API | JSON-based API for programmatic access |
| Data Preprocessing | Automatic scaling and sequence generation |
| Model Comparison | Compare predictions across different models |

---

## Model Architecture

### LSTM v1 (Standard Model)

```
Input Layer: (50 timesteps, 1 feature)
    |
    v
+----------------------------------+
|  LSTM Layer 1                    |
|  - Units: 128                    |
|  - Return Sequences: True        |
|  - Activation: tanh              |
+----------------------------------+
|  Batch Normalization             |
+----------------------------------+
|  Dropout: 0.5 (50%)              |
+----------------------------------+
    |
    v
+----------------------------------+
|  LSTM Layer 2                    |
|  - Units: 128                    |
|  - Return Sequences: False       |
|  - Activation: tanh              |
+----------------------------------+
|  Batch Normalization             |
+----------------------------------+
|  Dropout: 0.5 (50%)              |
+----------------------------------+
    |
    v
+----------------------------------+
|  Dense Layer                     |
|  - Units: 64                     |
|  - Activation: ReLU              |
+----------------------------------+
    |
    v
+----------------------------------+
|  Output Layer                    |
|  - Units: 1                      |
|  - Activation: Linear            |
+----------------------------------+
    |
    v
Output: Predicted Price
```

### LSTM v2 (Recommended - Better Generalization)

Same architecture as LSTM v1, but with **0.7 (70%) dropout** rate to reduce overfitting and improve generalization on unseen data.

### GRU v1 (Faster Inference)

```
Input Layer: (50 timesteps, 1 feature)
    |
    v
+----------------------------------+
|  GRU Layer 1                     |
|  - Units: 64                     |
|  - Return Sequences: True        |
+----------------------------------+
|  Batch Normalization             |
+----------------------------------+
|  Dropout: 0.5 (50%)              |
+----------------------------------+
    |
    v
+----------------------------------+
|  GRU Layer 2                     |
|  - Units: 64                     |
|  - Return Sequences: False       |
+----------------------------------+
|  Batch Normalization             |
+----------------------------------+
|  Dropout: 0.5 (50%)              |
+----------------------------------+
    |
    v
+----------------------------------+
|  Dense Layer (64 units, ReLU)    |
+----------------------------------+
    |
    v
+----------------------------------+
|  Output Layer (1 unit, Linear)   |
+----------------------------------+
```

---

## Model Performance

### Training Configuration

| Parameter | LSTM v1 | LSTM v2 | GRU v1 |
|-----------|---------|---------|--------|
| **Architecture** | 2-Layer LSTM | 2-Layer LSTM | 2-Layer GRU |
| **Units per Layer** | 128 | 128 | 64 |
| **Dropout Rate** | 0.5 (50%) | 0.7 (70%) | 0.5 (50%) |
| **Batch Size** | 64 | 64 | 32 |
| **Epochs** | 100 | 100 | 100 |
| **Optimizer** | Adam | Adam | Adam |
| **Learning Rate** | 0.001 | 0.001 | 0.001 |
| **Loss Function** | MSE | MSE | MSE |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error - measures average prediction error magnitude |
| **MAE** | Mean Absolute Error - average absolute difference between predicted and actual |
| **MAPE** | Mean Absolute Percentage Error - percentage-based error metric |
| **R-squared** | Coefficient of determination - indicates how well predictions fit actual data |

### Model Comparison

| Model | Best For | Trade-offs |
|-------|----------|------------|
| **LSTM v2** | General use, production | Best generalization, slightly slower |
| **LSTM v1** | Pattern-sensitive predictions | May overfit on training data |
| **GRU v1** | Fast inference, real-time apps | Fewer parameters, may miss subtle patterns |

---

## How It Works

### 1. Data Collection and Preprocessing

```
Raw Data (1-minute intervals)
         |
         v
+----------------------------------+
|  Resample to Daily Data          |
|  - Open: First value             |
|  - High: Maximum                 |
|  - Low: Minimum                  |
|  - Close: Last value             |
|  - Volume: Sum                   |
+----------------------------------+
         |
         v
+----------------------------------+
|  Extract Close Price             |
|  (Primary feature)               |
+----------------------------------+
         |
         v
+----------------------------------+
|  MinMax Normalization            |
|  Scale to [0, 1] range           |
+----------------------------------+
```

### 2. Sequence Generation

The model uses a **sliding window approach** with 50-day lookback:

```
Day 1   Day 2   Day 3   ...  Day 50  -->  Predict Day 51
 |       |       |            |                |
 +-------+-------+------------+                |
            Input (X)                      Target (y)
```

### 3. Training Process

```
+----------------------------------+
|  Forward Pass                    |
|  - Input: 50 days of prices      |
|  - Output: Predicted price       |
+----------------------------------+
         |
         v
+----------------------------------+
|  Calculate Loss (MSE)            |
|  - Compare prediction vs actual  |
+----------------------------------+
         |
         v
+----------------------------------+
|  Backpropagation                 |
|  - Update weights                |
|  - Adam optimizer                |
+----------------------------------+
         |
         v
+----------------------------------+
|  Callbacks                       |
|  - ReduceLROnPlateau             |
|  - ModelCheckpoint               |
+----------------------------------+
```

### 4. Prediction Pipeline

```
User Input (50 daily prices)
         |
         v
+----------------------------------+
|  Validate Input                  |
|  - Check count = 50              |
|  - Parse numeric values          |
+----------------------------------+
         |
         v
+----------------------------------+
|  Scale Input                     |
|  - Apply MinMaxScaler            |
|  - Transform to [0, 1]           |
+----------------------------------+
         |
         v
+----------------------------------+
|  Reshape for Model               |
|  - Shape: (1, 50, 1)             |
|  - [batch, timesteps, features]  |
+----------------------------------+
         |
         v
+----------------------------------+
|  Model Inference                 |
|  - Forward pass through network  |
|  - Get scaled prediction         |
+----------------------------------+
         |
         v
+----------------------------------+
|  Return Prediction               |
|  - Next day's predicted price    |
+----------------------------------+
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- 4GB+ RAM recommended
- GPU optional (for training)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/bitcoin-price-predictor.git
cd bitcoin-price-predictor

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.0
pandas
numpy
scikit-learn
matplotlib
seaborn
Flask>=2.0
h5py
Werkzeug>=2.0
joblib
```

---

## Usage

### Web Interface

1. Start the Flask server:
```bash
python api.py
```

2. Open browser at `http://localhost:5000`

3. Select a model, enter 50 daily prices (or load sample data), and click "Predict"

### Command Line Interface

```bash
# List available models
python predict.py --list-models

# Predict using recent historical data
python predict.py --model lstm_v2 --use-recent

# Predict using custom prices
python predict.py --model lstm_v2 --prices "42000,42100,42200,..."
```

### Python API

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model('notebooks/saved_model/model_lstm_l2_v2.keras')

# Prepare input (50 days of prices, scaled)
prices = np.array([...])  # 50 values
scaled = scaler.transform(prices.reshape(-1, 1))
X = scaled.reshape(1, 50, 1)

# Predict
prediction = model.predict(X)
```

---

## API Reference

### Endpoints

#### `GET /`
Renders the web interface.

#### `POST /predict`
Make a price prediction.

**Request Body:**
```json
{
  "prices": [42000, 42100, 42200, ...],
  "model": "lstm_v2"
}
```

**Response:**
```json
{
  "success": true,
  "predicted_price": 43250.50,
  "model_used": "lstm_v2",
  "input_prices_count": 50,
  "last_input_price": 42900.00,
  "price_change": 350.50,
  "price_change_pct": 0.82
}
```

#### `GET /api/recent-prices`
Get recent historical prices.

**Response:**
```json
{
  "prices": [42000, 42100, ...],
  "dates": ["2024-01-01", "2024-01-02", ...]
}
```

#### `GET /api/models`
List available models.

#### `GET /health`
Health check endpoint.

---

## Project Structure

```
bitcoin-price-predictor/
|
+-- api.py                    # Flask web application
+-- predict.py                # CLI prediction script
+-- train.py                  # Training script
+-- evaluate.py               # Model evaluation script
+-- requirements.txt          # Python dependencies
+-- readme.MD                 # This file
|
+-- data/
|   +-- btcusd_1-min_data.csv     # Raw data (1-minute intervals)
|   +-- processed/
|       +-- processed_data.csv     # Cleaned and preprocessed data
|
+-- notebooks/
|   +-- bitcoin-1d-price-prediction-lstm.ipynb  # Training notebook
|   +-- saved_model/
|       +-- lstm_l2_v1.keras       # LSTM v1 model
|       +-- model_lstm_l2_v2.keras # LSTM v2 model (recommended)
|       +-- gru_l2_v1.keras        # GRU v1 model
|
+-- src/
|   +-- __init__.py
|   +-- data.py               # Data loading and preprocessing
|   +-- model.py              # Model definitions
|
+-- templates/
|   +-- index.html            # Web interface template
|
+-- tests/                    # Unit tests
```

---

## Technical Details

### Why LSTM for Price Prediction?

1. **Memory Cells**: LSTM networks have special memory cells that can store information for long periods, making them ideal for time-series data.

2. **Gate Mechanisms**: Three gates (input, forget, output) control information flow:
   - **Forget Gate**: Decides what information to discard
   - **Input Gate**: Decides what new information to store
   - **Output Gate**: Decides what to output

3. **Gradient Flow**: LSTM architecture prevents vanishing gradient problem, allowing learning from long sequences.

### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | More (3 gates) | Fewer (2 gates) |
| Memory | Separate cell state | Combined hidden state |
| Training Speed | Slower | Faster |
| Long Dependencies | Better | Good |
| Use Case | Complex patterns | Simpler patterns, speed |

### Data Normalization

We use MinMaxScaler to normalize prices to [0, 1] range:

```
scaled_value = (value - min) / (max - min)
```

This helps the neural network train more efficiently by ensuring all inputs are on a similar scale.

### Lookback Window

The 50-day lookback window was chosen based on:
- Sufficient historical context for pattern recognition
- Balance between memory requirements and performance
- Common practice in financial time-series forecasting

---

## Limitations

1. **Market Volatility**: Cryptocurrency markets are highly volatile and unpredictable. No model can accurately predict sudden market movements.

2. **Historical Patterns**: The model assumes future patterns will resemble historical ones, which may not always be true.

3. **External Factors**: The model does not account for:
   - News events
   - Regulatory changes
   - Market sentiment
   - Macroeconomic factors

4. **Not Financial Advice**: Predictions should NOT be used as the sole basis for investment decisions.

---

## License

MIT License - You are free to use, modify, and distribute this project.

---

## Acknowledgments

- TensorFlow/Keras for deep learning framework
- Flask for web application framework
- Scikit-learn for data preprocessing
- The open-source community for various tools and libraries

---

**Disclaimer**: This project is for educational purposes only. Cryptocurrency trading involves substantial risk. Always do your own research before making investment decisions.

