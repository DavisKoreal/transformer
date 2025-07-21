#!/bin/bash

# This script applies changes to the transformer repository for XAUUSD time series forecasting.
# It updates transformer.py, .gitignore, requirements.txt, and adds a README.md.

# Exit on error
set -e

# Directory containing this script
DIR=$(pwd)

# Step 1: Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not activated. Activating $DIR/venv..."
    if [ ! -d "$DIR/venv" ]; then
        echo "Error: Virtual environment not found at $DIR/venv"
        exit 1
    fi
    source "$DIR/venv/bin/activate"
else
    echo "Virtual environment already activated: $VIRTUAL_ENV"
fi

# Step 2: Install required packages
echo "Installing required packages..."
pip install torch numpy pandas scikit-learn ta

# Step 3: Update requirements.txt
echo "Generating requirements.txt..."
pip freeze > "$DIR/requirements.txt"
echo "Updated requirements.txt with current dependencies"

# Step 4: Update .gitignore
echo "Updating .gitignore..."
cat << EOF > "$DIR/.gitignore"
# Python virtual environment
venv/

# Python cache files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
dist/
*.egg-info/

# macOS
.DS_Store

# IDEs and editors
.vscode/
.idea/

# Jupyter Notebook
.ipynb_checkpoints/

# Logs and temporary files
*.log
*.tmp

# Data files
*.csv
*.parquet
*.h5
EOF
echo "Updated .gitignore with Python and data file ignores"

# Step 5: Update transformer.py
echo "Updating transformer.py for XAUUSD time series forecasting..."
cat << 'EOF' > "$DIR/transformer.py"
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device() -> torch.device:
    """Determines the available device for computation.

    Returns:
        torch.device: CUDA if available, else CPU.
    """
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device

def scaled_dot_product(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs scaled dot-product attention computation.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, head_dim).
        mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Output values and attention weights.

    Raises:
        ValueError: If input tensors have incompatible shapes.
    """
    if query.size()[-1] != key.size()[-1] or key.size()[-1] != value.size()[-1]:
        logging.error("Incompatible dimensions in scaled_dot_product")
        raise ValueError("Query, key, and value must have the same head dimension")
    
    d_k: int = query.size()[-1]
    scaled: torch.Tensor = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention: torch.Tensor = F.softmax(scaled, dim=-1)
    values: torch.Tensor = torch.matmul(attention, value)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int):
        """Initializes positional encoding for time series inputs.

        Args:
            d_model (int): Dimension of the model embeddings.
            max_sequence_length (int): Maximum sequence length for encoding.
        """
        super().__init__()
        self.max_sequence_length: int = max_sequence_length
        self.d_model: int = d_model

    def forward(self) -> torch.Tensor:
        """Generates positional encoding tensor.

        Returns:
            torch.Tensor: Positional encoding tensor of shape (max_sequence_length, d_model).
        """
        even_indices: torch.Tensor = torch.arange(0, self.d_model, 2).float()
        denominator: torch.Tensor = torch.pow(10000, even_indices / self.d_model)
        positions: torch.Tensor = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_pe: torch.Tensor = torch.sin(positions / denominator)
        odd_pe: torch.Tensor = torch.cos(positions / denominator)
        stacked: torch.Tensor = torch.stack([even_pe, odd_pe], dim=2)
        pe: torch.Tensor = torch.flatten(stacked, start_dim=1, end_dim=2)
        return pe

class TimeSeriesEmbedding(nn.Module):
    NUM_FEATURES: int = 10  # Number of features per time step (OHLCV + indicators + USD_Index)

    def __init__(self, d_model: int, max_sequence_length: int, num_features: int = NUM_FEATURES):
        """Initializes embedding module for time series data.

        Args:
            d_model (int): Dimension of the model embeddings.
            max_sequence_length (int): Maximum sequence length, adjustable for input series.
            num_features (int): Number of features per time step (default: 10).

        Raises:
            ValueError: If num_features is not 10.
        """
        super().__init__()
        if num_features != self.NUM_FEATURES:
            logging.error(f"Expected {self.NUM_FEATURES} features, got {num_features}")
            raise ValueError(f"Number of features must be {self.NUM_FEATURES}")
        
        self.max_sequence_length: int = max_sequence_length
        self.d_model: int = d_model
        self.num_features: int = num_features
        self.projection: nn.Linear = nn.Linear(num_features, d_model)
        self.position_encoder: PositionalEncoding = PositionalEncoding(d_model, max_sequence_length)
        self.dropout: nn.Dropout = nn.Dropout(p=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds a batch of time series data with positional encoding.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, max_sequence_length, d_model).

        Raises:
            ValueError: If input sequence length exceeds max_sequence_length or feature count is incorrect.
        """
        batch_size, seq_len, num_features = x.size()
        if num_features != self.num_features:
            logging.error(f"Expected {self.num_features} features, got {num_features}")
            raise ValueError(f"Input must have {self.num_features} features")
        if seq_len > self.max_sequence_length:
            logging.error(f"Sequence length {seq_len} exceeds max_sequence_length {self.max_sequence_length}")
            raise ValueError(f"Sequence length must not exceed {self.max_sequence_length}")

        # Pad if sequence is too short
        if seq_len < self.max_sequence_length:
            padding: torch.Tensor = torch.zeros(batch_size, self.max_sequence_length - seq_len, num_features).to(x.device)
            x = torch.cat([x, padding], dim=1)
        
        x = self.projection(x)
        pos: torch.Tensor = self.position_encoder().to(x.device)
        x = self.dropout(x + pos)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """Initializes multi-head attention module.

        Args:
            d_model (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        super().__init__()
        if d_model % num_heads != 0:
            logging.error(f"d_model {d_model} must be divisible by num_heads {num_heads}")
            raise ValueError("d_model must be divisible by num_heads")
        
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.qkv_layer: nn.Linear = nn.Linear(d_model, 3 * d_model)
        self.linear_layer: nn.Linear = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs multi-head attention computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, sequence_length, d_model = x.size()
        qkv: torch.Tensor = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        query, key, value = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(query, key, value, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out: torch.Tensor = self.linear_layer(values)
        return out

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape: list[int], eps: float = 1e-5):
        """Initializes layer normalization module.

        Args:
            parameters_shape (list[int]): Shape of the input tensor for normalization.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.parameters_shape: list[int] = parameters_shape
        self.eps: float = eps
        self.gamma: nn.Parameter = nn.Parameter(torch.ones(parameters_shape))
        self.beta: nn.Parameter = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization to the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        dims: list[int] = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean: torch.Tensor = inputs.mean(dim=dims, keepdim=True)
        var: torch.Tensor = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std: torch.Tensor = (var + self.eps).sqrt()
        y: torch.Tensor = (inputs - mean) / std
        out: torch.Tensor = self.gamma * y + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        """Initializes position-wise feed-forward network.

        Args:
            d_model (int): Dimension of the model embeddings.
            hidden (int): Dimension of the hidden layer.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.linear1: nn.Linear = nn.Linear(d_model, hidden)
        self.linear2: nn.Linear = nn.Linear(hidden, d_model)
        self.relu: nn.ReLU = nn.ReLU()
        self.dropout: nn.Dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        """Initializes a single encoder layer.

        Args:
            d_model (int): Dimension of the model embeddings.
            ffn_hidden (int): Dimension of the feed-forward hidden layer.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.attention: MultiHeadAttention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1: LayerNormalization = LayerNormalization(parameters_shape=[d_model])
        self.dropout1: nn.Dropout = nn.Dropout(p=drop_prob)
        self.ffn: PositionwiseFeedForward = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2: LayerNormalization = LayerNormalization(parameters_shape=[d_model])
        self.dropout2: nn.Dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor, self_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Processes input through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            self_attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        residual_x: torch.Tensor = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> torch.Tensor:
        """Processes input through a sequence of encoder layers.

        Args:
            inputs (Tuple[torch.Tensor, Optional[torch.Tensor]]): Input tensor and optional attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int
    ):
        """Initializes the encoder module for time series data.

        Args:
            d_model (int): Dimension of the model embeddings.
            ffn_hidden (int): Dimension of the feed-forward hidden layer.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability.
            num_layers (int): Number of encoder layers.
            max_sequence_length (int): Maximum sequence length, adjustable for input series.
        """
        super().__init__()
        self.time_series_embedding: TimeSeriesEmbedding = TimeSeriesEmbedding(d_model, max_sequence_length)
        self.layers: SequentialEncoder = SequentialEncoder(
            *[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, self_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Processes time series input through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            self_attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, max_sequence_length, d_model).
        """
        x = self.time_series_embedding(x)
        x = self.layers(x, self_attention_mask)
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """Initializes multi-head cross-attention module.

        Args:
            d_model (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        super().__init__()
        if d_model % num_heads != 0:
            logging.error(f"d_model {d_model} must be divisible by num_heads {num_heads}")
            raise ValueError("d_model must be divisible by num_heads")
        
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.kv_layer: nn.Linear = nn.Linear(d_model, 2 * d_model)
        self.q_layer: nn.Linear = nn.Linear(d_model, d_model)
        self.linear_layer: nn.Linear = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs multi-head cross-attention computation.

        Args:
            x (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            y (torch.Tensor): Decoder input tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, sequence_length, d_model = x.size()
        kv: torch.Tensor = self.kv_layer(x)
        q: torch.Tensor = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out: torch.Tensor = self.linear_layer(values)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        """Initializes a single decoder layer.

        Args:
            d_model (int): Dimension of the model embeddings.
            ffn_hidden (int): Dimension of the feed-forward hidden layer.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.self_attention: MultiHeadAttention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1: LayerNormalization = LayerNormalization(parameters_shape=[d_model])
        self.dropout1: nn.Dropout = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention: MultiHeadCrossAttention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2: LayerNormalization = LayerNormalization(parameters_shape=[d_model])
        self.dropout2: nn.Dropout = nn.Dropout(p=drop_prob)
        self.ffn: PositionwiseFeedForward = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3: LayerNormalization = LayerNormalization(parameters_shape=[d_model])
        self.dropout3: nn.Dropout = nn.Dropout(p=drop_prob)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Processes input through the decoder layer.

        Args:
            x (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            y (torch.Tensor): Decoder input tensor of shape (batch_size, seq_len, d_model).
            self_attention_mask (Optional[torch.Tensor]): Self-attention mask of shape (batch_size, 1, seq_len, seq_len).
            cross_attention_mask (Optional[torch.Tensor]): Cross-attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        residual_y: torch.Tensor = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + residual_y)
        residual_y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + residual_y)
        residual_y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + residual_y)
        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]) -> torch.Tensor:
        """Processes input through a sequence of decoder layers.

        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]):
                Encoder output, decoder input, self-attention mask, and cross-attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int
    ):
        """Initializes the decoder module for time series data.

        Args:
            d_model (int): Dimension of the model embeddings.
            ffn_hidden (int): Dimension of the feed-forward hidden layer.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability.
            num_layers (int): Number of decoder layers.
            max_sequence_length (int): Maximum sequence length for decoder input.
        """
        super().__init__()
        self.time_series_embedding: TimeSeriesEmbedding = TimeSeriesEmbedding(d_model, max_sequence_length)
        self.layers: SequentialDecoder = SequentialDecoder(
            *[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Processes time series input through the decoder.

        Args:
            x (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            y (torch.Tensor): Target time series tensor of shape (batch_size, seq_len, num_features).
            self_attention_mask (Optional[torch.Tensor]): Self-attention mask of shape (batch_size, 1, seq_len, seq_len).
            cross_attention_mask (Optional[torch.Tensor]): Cross-attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, max_sequence_length, d_model).
        """
        y = self.time_series_embedding(y)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int,
        forecast_horizon: int
    ):
        """Initializes the transformer model for XAUUSD time series forecasting.

        Args:
            d_model (int): Dimension of the model embeddings.
            ffn_hidden (int): Dimension of the feed-forward hidden layer.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability.
            num_layers (int): Number of encoder and decoder layers.
            max_sequence_length (int): Maximum input sequence length, adjustable for input series.
            forecast_horizon (int): Number of future time steps to predict.

        Raises:
            ValueError: If max_sequence_length or forecast_horizon is less than 1.
        """
        super().__init__()
        if max_sequence_length < 1 or forecast_horizon < 1:
            logging.error(f"Invalid max_sequence_length {max_sequence_length} or forecast_horizon {forecast_horizon}")
            raise ValueError("max_sequence_length and forecast_horizon must be at least 1")
        
        self.encoder: Encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length)
        self.decoder: Decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, forecast_horizon)
        self.linear: nn.Linear = nn.Linear(d_model, 1)  # Predict closing price
        self.device: torch.device = get_device()

    def create_causal_mask(self, size: int) -> torch.Tensor:
        """Creates a causal mask for the decoder to prevent attending to future time steps.

        Args:
            size (int): Size of the sequence (forecast_horizon).

        Returns:
            torch.Tensor: Causal mask of shape (1, 1, size, size).
        """
        mask: torch.Tensor = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).to(self.device)

    def create_padding_mask(self, seq_len: int, max_seq_len: int) -> torch.Tensor:
        """Creates a padding mask for sequences shorter than max_sequence_length.

        Args:
            seq_len (int): Actual sequence length.
            max_seq_len (int): Maximum sequence length.

        Returns:
            torch.Tensor: Padding mask of shape (1, 1, max_seq_len, max_seq_len).
        """
        mask: torch.Tensor = torch.zeros(max_seq_len, max_seq_len)
        if seq_len < max_seq_len:
            mask[seq_len:, :] = float('-inf')
            mask[:, seq_len:] = float('-inf')
        return mask.unsqueeze(0).unsqueeze(0).to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        encoder_self_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Processes time series input to predict future closing prices.

        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, num_features).
            y (torch.Tensor): Target time series of shape (batch_size, forecast_horizon, num_features).
            encoder_self_attention_mask (Optional[torch.Tensor]): Encoder attention mask.

        Returns:
            torch.Tensor: Predicted closing prices of shape (batch_size, forecast_horizon).

        Raises:
            ValueError: If input shapes are invalid.
        """
        batch_size, seq_len, num_features = x.size()
        if num_features != TimeSeriesEmbedding.NUM_FEATURES:
            logging.error(f"Expected {TimeSeriesEmbedding.NUM_FEATURES} features, got {num_features}")
            raise ValueError(f"Input must have {TimeSeriesEmbedding.NUM_FEATURES} features")
        
        # Apply padding mask if sequence is shorter than max_sequence_length
        if seq_len < self.encoder.time_series_embedding.max_sequence_length:
            encoder_self_attention_mask = self.create_padding_mask(seq_len, self.encoder.time_series_embedding.max_sequence_length)
        
        decoder_self_attention_mask: torch.Tensor = self.create_causal_mask(y.size(1))
        x = self.encoder(x, encoder_self_attention_mask)
        out = self.decoder(x, y, decoder_self_attention_mask, None)
        out = self.linear(out).squeeze(-1)  # Shape: (batch_size, forecast_horizon)
        return out
EOF
echo "Updated transformer.py for XAUUSD time series forecasting"

# Step 6: Create README.md
echo "Creating README.md..."
cat << EOF > "$DIR/README.md"
# XAUUSD Time Series Forecasting

This project implements a Transformer model for forecasting XAUUSD (Gold/USD) prices using a time series of 10 features per time step.

## Setup

1. **Activate Virtual Environment**:
   \`\`\`bash
   source venv/bin/activate
   \`\`\`

2. **Install Dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Prepare Data**:
   - Obtain XAUUSD OHLCV data and USD Index data (e.g., from Yahoo Finance or MetaTrader).
   - Compute features: Open, High, Low, Close, Volume, SMA, RSI, MACD, Bollinger Band Width, USD Index.
   - Normalize features using MinMaxScaler.

4. **Run the Model**:
   \`\`\`python
   import torch
   from transformer import Transformer

   model = Transformer(d_model=64, ffn_hidden=256, num_heads=4, drop_prob=0.1, num_layers=2, max_sequence_length=100, forecast_horizon=10)
   model.to(get_device())
   x = torch.tensor(feature_data_normalized[:100], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 100, 10)
   y = torch.tensor(feature_data_normalized[100:110, 3:4], dtype=torch.float32).unsqueeze(0)  # Predict Close
   output = model(x, y)
   \`\`\`

## Dependencies
- torch
- numpy
- pandas
- scikit-learn
- ta

## Notes
- The `max_sequence_length` parameter is adjustable for variable input sequence lengths.
- The model predicts the closing price for the specified forecast horizon.
- Use Git to track changes and create feature branches for modifications.
EOF
echo "Created README.md with setup and usage instructions"

# Step 7: Display completion message
echo "Changes applied successfully! transformer.py, .gitignore, requirements.txt, and README.md have been updated."
echo "To stage changes for Git, run:"
echo "  git add transformer.py .gitignore requirements.txt README.md"
echo "To deactivate the virtual environment, run 'deactivate'."