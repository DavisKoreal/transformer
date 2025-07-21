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
   - Obtain XAUUSD OHLCV data and USD Index data (e.g., from Yahoo Finance or MetaTrader).
   - Compute features: Open, High, Low, Close, Volume, SMA, RSI, MACD, Bollinger Band Width, USD Index.
   - Normalize features using MinMaxScaler.

4. **Run the Model**:
   ```python
   import torch
   from transformer import Transformer

   model = Transformer(d_model=64, ffn_hidden=256, num_heads=4, drop_prob=0.1, num_layers=2, max_sequence_length=100, forecast_horizon=10)
   model.to(get_device())
   x = torch.tensor(feature_data_normalized[:100], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 100, 10)
   y = torch.tensor(feature_data_normalized[100:110, 3:4], dtype=torch.float32).unsqueeze(0)  # Predict Close
   output = model(x, y)
   ```

## Dependencies
- torch
- numpy
- pandas
- scikit-learn
- ta

## Notes
- The  parameter is adjustable for variable input sequence lengths.
- The model predicts the closing price for the specified forecast horizon.
- Use Git to track changes and create feature branches for modifications.
