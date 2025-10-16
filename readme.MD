# 📊 BitCoin‑Price‑Predictor  
A quick‑start guide for the “Bitcoin Price Predictor” project – an end‑to‑end **LSTM‑based** time‑series forecasting tool.

---

## 📚️ Overview  

*The goal:*  
> Build a reusable Python package that (1) pulls raw Bitcoin price data, (2) engineers useful features,  
> (3) trains a small LSTM model (TensorFlow/Keras or PyTorch), and (4) serves predictions via a tiny API / CLI.

*Why this matters:*  
- Rapid, repeatable experiments for crypto‑trading analytics.  
- Easy to plug into other projects – just import the `Predictor` class.  
- Open‑source: share, fork and extend.

---

## 📂 Contents

| Section | What you’ll find |
|---|---|
| **Prerequisites** | Libraries, Python version and GPU/CPU requirements. |
| **Installation** | How to get the repo & install. |
| **Data preparation** | Download, clean and split data. |
| **Training** | Run the training script – hyper‑parameter tuning, checkpoints, logs. |
| **Evaluation** | How to compute MAE / RMSE and plot results. |
| **Inference** | Two ways: script or REST API. |
| **Project Structure** | Where is what – code, notebooks, tests. |
| **Contributing** | Pull‑requests, coding style, docs. |
| **License** | MIT. |

---

## 📁 Prerequisites  

```bash
# Create virtual env (recommended)
python3 -m venv .venv   # or conda create -n price-pred python=3.10
source .venv/bin/activate

# Install dependencies  
pip install -r requirements.txt
```

Dependencies (requirements.txt)

```text
python==3.10
tensorflow>=2.0          # or pytorch
pandas
numpy
scikit‑learn
matplotlib
seaborn
```

---

## 📂 Installation

```bash
# 1. Clone the repo  
git clone https://github.com/your‑org/bitcoin‑price‑predictor.git

# 2. Install  
cd bitcoin-price-predictor
pip install -r requirements.txt
```

---

## 📃 Data preparation  

1. **Download** the dataset (e.g., Kaggle or CoinGecko).  
```python
import pandas as pd
df = pd.read_csv("bitcoin_raw.csv")     
```

1. **Clean** – drop nulls or impute.
```python
df = df.dropna()
df["Close"] = pd.to_datetime(df["Close"])
```

1. **Feature‑engineer** – rolling windows, EMA, RSI…
```python
df["MA30"] = df.Close.rolling(30).mean()
# … add more as you wish
```

1. **Save** the processed data for training.
```python
df.to_csv("processed/bitcoin_train.csv",index=False)
```

---

## 📄 Training  

```bash
python train.py \
  --data processed/bitcoin_train.csv \
  --epochs 20 \
  --batch 32 \
  --units 64
```

The script will:

- Load data
- Build the LSTM network (1–2 layers, 64 units per layer)
- Train with Adam optimizer  
- Save the best model checkpoint (`model.h5`)

---

## 📅 Evaluation  

```bash
python evaluate.py --model model.h5 --test processed/test.csv
```

You’ll see:

- RMSE, MAE tables
- Plot of real vs predicted

---

## 📆 Inference  

**CLI**

```bash
python predict.py --model model.h5 --date 2024‑10‑01
```

**REST (Flask)** – see `api/server.py` for a tiny endpoint.

---

## 📊 Project Structure  

```
bitcoin-price-predictor/
├─ data/                     # raw & processed csv files
├─ notebooks/                 # Jupyter for quick experiments
├─ src/
│   ├─ __init__.py          # expose train / predict helper
│   ├─ data.py               # load and preprocess
│   ├─ model.py              # LSTM definition, train, evaluate
├─ tests/                    # unit tests for utils
├─ train.py                  # CLI training entry point
├─ evaluate.py               # eval script
└─ predict.py                # inference
```

---

## 📈 Contributing  

1. Fork & create a branch.  
2. Add tests for any new change.  
3. Follow PEP‑8 style & run `flake8` before PR.  
4. Documentation: add function docstring + `README.md`.  

---

## 📉 License  

MIT – you’re free to use, modify and distribute.

