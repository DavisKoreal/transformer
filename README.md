# XAUUSD Time Series Forecasting

This project implements a Transformer model for forecasting XAUUSD (Gold/USD) prices using a time series of 10 features per time step.

## Setup

1. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Run the data preparation script to fetch XAUUSD and USD Index data, compute features, and save to CSV:
     ```bash
     python data_preparation.py --start-date 2020-01-01 --end-date 2025-07-22 --interval 1d
     ```
   - Features: Open, High, Low, Close, Volume, SMA, RSI, MACD, Bollinger Band Width, USD Index.
   - Output: `xauusd_features.csv` (normalized data saved in the project directory).
   - Use `--force` to regenerate the CSV if it exists.

4. **Run the Model**:
   ```python
   import torch
   import numpy as np
   from transformer import Transformer

   # Load prepared data
   data = np.loadtxt('xauusd_features.csv', delimiter=',', skiprows=1)
   model = Transformer(d_model=64, ffn_hidden=256, num_heads=4, drop_prob=0.1, num_layers=2, max_sequence_length=100, forecast_horizon=10)
   model.to(model.get_device())
   x = torch.tensor(data[:100], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 100, 10)
   y = torch.tensor(data[100:110, 3:4], dtype=torch.float32).unsqueeze(0)  # Predict Close
   output = model(x, y)
   ```

## Dependencies
- torch
- numpy
- pandas
- scikit-learn
- ta
- yfinance

## Notes
- The `max_sequence_length` parameter is adjustable for variable input sequence lengths.
- The model predicts the closing price for the specified forecast horizon.
- Use Git to track changes and create feature branches for modifications.
