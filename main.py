import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from typing import Tuple
from pathlib import Path
from transformer import Transformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str, max_sequence_length: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Loads and prepares data for training and testing.

    Args:
        file_path (str): Path to xauusd_features.csv.
        max_sequence_length (int): Length of input sequences.
        forecast_horizon (int): Number of time steps to predict.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Training and testing data arrays.

    Raises:
        FileNotFoundError: If xauusd_features.csv is missing.
        ValueError: If data is insufficient, has incorrect columns, or contains non-numeric values.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logging.error(f"Data file {file_path} not found")
        raise FileNotFoundError(f"Please run data_preparation.py to generate {file_path}")

    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'MACD', 'BB_Width', 'USD_Index']
    try:
        # Load CSV with date index and select expected columns
        data = pd.read_csv(file_path, index_col=0)
        # Verify all expected columns are present
        if not all(col in data.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in data.columns]
            logging.error(f"Missing columns in CSV: {missing}")
            raise ValueError(f"CSV must contain columns: {expected_columns}")
        data = data[expected_columns].astype(np.float64).values  # Shape: (num_timesteps, 10)
        logging.info(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        logging.error(f"Failed to load CSV: {e}")
        raise ValueError(f"Failed to load {file_path}: {e}") from e

    if data.shape[1] != 10:
        logging.error(f"Expected 10 features, got {data.shape[1]}")
        raise ValueError("Data must have 10 features (OHLCV, SMA, RSI, MACD, BB_Width, USD_Index)")
    
    if data.shape[0] < max_sequence_length + forecast_horizon:
        logging.error(f"Data too short: {data.shape[0]} timesteps, need at least {max_sequence_length + forecast_horizon}")
        raise ValueError("Insufficient data for sequence length and forecast horizon")

    # Split into train (80%) and test (20%)
    train_size = int(0.8 * data.shape[0])  # Corrected line
    train_data = data[:train_size]
    test_data = data[train_size - max_sequence_length:]  # Include overlap for sequences
    logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    return train_data, test_data

def create_sequences(
    data: np.ndarray,
    max_sequence_length: int,
    forecast_horizon: int,
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates input and target sequences for training or testing.

    Args:
        data (np.ndarray): Input data of shape (num_timesteps, 10).
        max_sequence_length (int): Length of input sequences.
        forecast_horizon (int): Number of time steps to predict.
        batch_size (int): Batch size for sequences.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input sequences (batch_size, seq_len, 10) and targets (batch_size, forecast_horizon, 1).

    Raises:
        ValueError: If data is insufficient for sequences.
    """
    if data.shape[0] < max_sequence_length + forecast_horizon:
        logging.error(f"Data too short: {data.shape[0]} timesteps for seq_len={max_sequence_length}, horizon={forecast_horizon}")
        raise ValueError("Insufficient data for sequences")

    x, y = [], []
    for i in range(0, data.shape[0] - max_sequence_length - forecast_horizon + 1, batch_size):
        batch_x = []
        batch_y = []
        for j in range(min(batch_size, data.shape[0] - max_sequence_length - forecast_horizon + 1 - i)):
            batch_x.append(data[i + j:i + j + max_sequence_length])
            batch_y.append(data[i + j + max_sequence_length:i + j + max_sequence_length + forecast_horizon, 3:4])  # Close price
        if batch_x:
            x.append(np.stack(batch_x))
            y.append(np.stack(batch_y))
    
    if not x:
        logging.error("No sequences generated")
        raise ValueError("Failed to generate sequences")
    
    x_tensor = torch.tensor(np.concatenate(x), dtype=torch.float32)
    y_tensor = torch.tensor(np.concatenate(y), dtype=torch.float32)
    logging.info(f"Created sequences: x shape={x_tensor.shape}, y shape={y_tensor.shape}")
    return x_tensor, y_tensor

def train_model(
    model: Transformer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float = 0.001
) -> None:
    """Trains the Transformer model.

    Args:
        model (Transformer): The Transformer model to train.
        train_x (torch.Tensor): Training input sequences of shape (batch_size, seq_len, 10).
        train_y (torch.Tensor): Training target sequences of shape (batch_size, forecast_horizon, 1).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.

    Raises:
        ValueError: If input shapes are invalid.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = model.device
    train_x, train_y = train_x.to(device), train_y.to(device)

    if train_x.size(2) != 10 or train_y.size(2) != 1:
        logging.error(f"Invalid input shapes: x={train_x.shape}, y={train_y.shape}")
        raise ValueError("Expected x with 10 features and y with 1 feature (Close)")

    logging.info(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, train_x.size(0), batch_size):
            batch_x = train_x[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]
            if batch_x.size(0) == 0:
                continue

            optimizer.zero_grad()
            output = model(batch_x, batch_y)
            loss = criterion(output, batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, (train_x.size(0) + batch_size - 1) // batch_size)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")

    logging.info("Training completed")
    torch.save(model.state_dict(), "transformer_model.pth")
    logging.info("Saved model to transformer_model.pth")

def evaluate_model(model: Transformer, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    """Evaluates the Transformer model on the test set.

    Args:
        model (Transformer): The trained Transformer model.
        test_x (torch.Tensor): Test input sequences of shape (batch_size, seq_len, 10).
        test_y (torch.Tensor): Test target sequences of shape (batch_size, forecast_horizon, 1).

    Returns:
        float: Mean Squared Error on the test set.

    Raises:
        ValueError: If input shapes are invalid.
    """
    model.eval()
    criterion = nn.MSELoss()
    device = model.device
    test_x, test_y = test_x.to(device), test_y.to(device)

    if test_x.size(2) != 10 or test_y.size(2) != 1:
        logging.error(f"Invalid input shapes: x={test_x.shape}, y={test_y.shape}")
        raise ValueError("Expected x with 10 features and y with 1 feature (Close)")

    with torch.no_grad():
        output = model(test_x, test_y)
        mse = criterion(output, test_y.squeeze(-1)).item()
    
    logging.info(f"Test MSE: {mse:.6f}")
    return mse

def main():
    """Main function to train and test the Transformer model."""
    parser = argparse.ArgumentParser(description="Train and test Transformer model for XAUUSD forecasting.")
    parser.add_argument('--data-path', default='xauusd_features.csv', help='Path to data CSV')
    parser.add_argument('--max-sequence-length', type=int, default=100, help='Input sequence length')
    parser.add_argument('--forecast-horizon', type=int, default=10, help='Number of time steps to predict')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--d-model', type=int, default=64, help='Dimension of model embeddings')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--ffn-hidden', type=int, default=256, help='Feed-forward hidden dimension')
    parser.add_argument('--drop-prob', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()

    try:
        # Load and prepare data
        train_data, test_data = load_data(args.data_path, args.max_sequence_length, args.forecast_horizon)
        train_x, train_y = create_sequences(train_data, args.max_sequence_length, args.forecast_horizon, args.batch_size)
        test_x, test_y = create_sequences(test_data, args.max_sequence_length, args.forecast_horizon, args.batch_size)

        # Initialize model
        model = Transformer(
            d_model=args.d_model,
            ffn_hidden=args.ffn_hidden,
            num_heads=args.num_heads,
            drop_prob=args.drop_prob,
            num_layers=args.num_layers,
            max_sequence_length=args.max_sequence_length,
            forecast_horizon=args.forecast_horizon
        )
        model.to(model.device)
        logging.info(f"Initialized Transformer model with {args.num_layers} layers, d_model={args.d_model}")

        # Train model
        train_model(model, train_x, train_y, args.epochs, args.batch_size, args.learning_rate)

        # Evaluate model
        mse = evaluate_model(model, test_x, test_y)
        logging.info(f"Final Test MSE: {mse:.6f}")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()