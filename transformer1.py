import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple

def get_device() -> torch.device:
    """
    Determines the available device for computation.
    Returns a CUDA device if available, otherwise returns CPU.
    
    Returns:
        device (torch.device): The selected device (CUDA or CPU).
    """
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device

def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs scaled dot-product attention computation.
    Takes query, key, and value tensors, and an optional mask.
    Computes attention scores, applies softmax, and returns weighted values.
    
    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, head_dim).
        mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).
    
    Returns:
        values (torch.Tensor): Output values after attention of shape (batch_size, num_heads, seq_len, head_dim).
        attention (torch.Tensor): Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
    """
    d_k: int = q.size()[-1]
    scaled: torch.Tensor = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention: torch.Tensor = F.softmax(scaled, dim=-1)
    values: torch.Tensor = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int):
        """
        Initializes positional encoding for transformer inputs.
        Computes sinusoidal encodings based on position and dimension.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            max_sequence_length (int): Maximum sequence length for encoding.
        """
        super().__init__()
        self.max_sequence_length: int = max_sequence_length
        self.d_model: int = d_model

    def forward(self) -> torch.Tensor:
        """
        Generates positional encoding tensor.
        
        Returns:
            PE (torch.Tensor): Positional encoding tensor of shape (max_sequence_length, d_model).
        """
        even_i: torch.Tensor = torch.arange(0, self.d_model, 2).float()
        denominator: torch.Tensor = torch.pow(10000, even_i / self.d_model)
        position: torch.Tensor = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE: torch.Tensor = torch.sin(position / denominator)
        odd_PE: torch.Tensor = torch.cos(position / denominator)
        stacked: torch.Tensor = torch.stack([even_PE, odd_PE], dim=2)
        PE: torch.Tensor = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    def __init__(
        self,
        max_sequence_length: int,
        d_model: int,
        language_to_index: Dict[str, int],
        START_TOKEN: str,
        END_TOKEN: str,
        PADDING_TOKEN: str
    ):
        """
        Initializes sentence embedding module.
        Converts tokenized sentences into embeddings with positional encoding.
        
        Args:
            max_sequence_length (int): Maximum length of input sequences.
            d_model (int): Dimension of the model embeddings.
            language_to_index (Dict[str, int]): Mapping of tokens to their indices.
            START_TOKEN (str): Token for start of sequence.
            END_TOKEN (str): Token for end of sequence.
            PADDING_TOKEN (str): Token for padding.
        """
        super().__init__()
        self.vocab_size: int = len(language_to_index)
        self.max_sequence_length: int = max_sequence_length
        self.embedding: nn.Embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index: Dict[str, int] = language_to_index
        self.position_encoder: PositionalEncoding = PositionalEncoding(d_model, max_sequence_length)
        self.dropout: nn.Dropout = nn.Dropout(p=0.1)
        self.START_TOKEN: str = START_TOKEN
        self.END_TOKEN: str = END_TOKEN
        self.PADDING_TOKEN: str = PADDING_TOKEN
    
    def batch_tokenize(
        self,
        batch: List[str],
        start_token: bool,
        end_token: bool
    ) -> torch.Tensor:
        """
        Tokenizes a batch of sentences into index tensors.
        
        Args:
            batch (List[str]): List of sentences to tokenize.
            start_token (bool): Whether to prepend start token.
            end_token (bool): Whether to append end token.
        
        Returns:
            tokenized (torch.Tensor): Tokenized batch of shape (batch_size, max_sequence_length).
        """
        def tokenize(sentence: str, start_token: bool, end_token: bool) -> torch.Tensor:
            sentence_word_indicies: List[int] = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)
        
        tokenized: List[torch.Tensor] = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
        tokenized_tensor: torch.Tensor = torch.stack(tokenized)
        return tokenized_tensor.to(get_device())
    
    def forward(self, x: List[str], start_token: bool, end_token: bool) -> torch.Tensor:
        """
        Embeds a batch of sentences with positional encoding.
        
        Args:
            x (List[str]): Batch of input sentences.
            start_token (bool): Whether to prepend start token.
            end_token (bool): Whether to append end token.
        
        Returns:
            x (torch.Tensor): Embedded sentences of shape (batch_size, max_sequence_length, d_model).
        """
        x: torch.Tensor = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos: torch.Tensor = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Initializes multi-head attention module.
        Splits input into multiple attention heads and processes them.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.qkv_layer: nn.Linear = nn.Linear(d_model, 3 * d_model)
        self.linear_layer: nn.Linear = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs multi-head attention computation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).
        
        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size: int
        sequence_length: int
        d_model: int
        batch_size, sequence_length, d_model = x.size()
        qkv: torch.Tensor = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q, k, v = qkv.chunk(3, dim=-1)
        values: torch.Tensor
        attention: torch.Tensor
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out: torch.Tensor = self.linear_layer(values)
        return out

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape: List[int], eps: float = 1e-5):
        """
        Initializes layer normalization module.
        Normalizes inputs across specified dimensions.
        
        Args:
            parameters_shape (List[int]): Shape of the input tensor for normalization.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.parameters_shape: List[int] = parameters_shape
        self.eps: float = eps
        self.gamma: nn.Parameter = nn.Parameter(torch.ones(parameters_shape))
        self.beta: nn.Parameter = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies layer normalization to the input tensor.
        
        Args:
            inputs (torch.Tensor): Input tensor to normalize.
        
        Returns:
            out (torch.Tensor): Normalized tensor of the same shape as input.
        """
        dims: List[int] = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean: torch.Tensor = inputs.mean(dim=dims, keepdim=True)
        var: torch.Tensor = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std: torch.Tensor = (var + self.eps).sqrt()
        y: torch.Tensor = (inputs - mean) / std
        out: torch.Tensor = self.gamma * y + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        """
        Initializes position-wise feed-forward network.
        Applies two linear transformations with a ReLU activation in between.
        
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
        """
        Processes input through the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        """
        Initializes a single encoder layer.
        Combines multi-head attention, normalization, and feed-forward network.
        
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

    def forward(self, x: torch.Tensor, self_attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Processes input through the encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            self_attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).
        
        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
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
        """
        Processes input through a sequence of encoder layers.
        
        Args:
            inputs (Tuple[torch.Tensor, Optional[torch.Tensor]]): Input tensor and optional attention mask.
        
        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
        """
        x: torch.Tensor
        self_attention_mask: Optional[torch.Tensor]
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
        max_sequence_length: int,
        language_to_index: Dict[str, int],
        START_TOKEN: str,
        END_TOKEN: str,
        PADDING_TOKEN: str
    ):
        """
        Initializes the encoder module.
        Combines sentence embedding with multiple encoder layers.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            ffn_hidden (int): Dimension of the feed-forward hidden layer.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability.
            num_layers (int): Number of encoder layers.
            max_sequence_length (int): Maximum sequence length.
            language_to_index (Dict[str, int]): Mapping of tokens to indices.
            START_TOKEN (str): Start token.
            END_TOKEN (str): End token.
            PADDING_TOKEN (str): Padding token.
        """
        super().__init__()
        self.sentence_embedding: SentenceEmbedding = SentenceEmbedding(
            max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN
        )
        self.layers: SequentialEncoder = SequentialEncoder(
            *[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: List[str],
        self_attention_mask: Optional[torch.Tensor],
        start_token: bool,
        end_token: bool
    ) -> torch.Tensor:
        """
        Processes input through the encoder.
        
        Args:
            x (List[str]): Batch of input sentences.
            self_attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).
            start_token (bool): Whether to prepend start token.
            end_token (bool): Whether to append end token.
        
        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, max_sequence_length, d_model).
        """
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Initializes multi-head cross-attention module.
        Processes queries from one sequence and keys/values from another.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.kv_layer: nn.Linear = nn.Linear(d_model, 2 * d_model)
        self.q_layer: nn.Linear = nn.Linear(d_model, d_model)
        self.linear_layer: nn.Linear = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs multi-head cross-attention computation.
        
        Args:
            x (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            y (torch.Tensor): Decoder input tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, seq_len, seq_len).
        
        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size: int
        sequence_length: int
        d_model: int
        batch_size, sequence_length, d_model = x.size()
        kv: torch.Tensor = self.kv_layer(x)
        q: torch.Tensor = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k: torch.Tensor
        v: torch.Tensor
        k, v = kv.chunk(2, dim=-1)
        values: torch.Tensor
        attention: torch.Tensor
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out: torch.Tensor = self.linear_layer(values)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        """
        Initializes a single decoder layer.
        Combines self-attention, cross-attention, and feed-forward network.
        
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
        self_attention_mask: Optional[torch.Tensor],
        cross_attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Processes input through the decoder layer.
        
        Args:
            x (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            y (torch.Tensor): Decoder input tensor of shape (batch_size, seq_len, d_model).
            self_attention_mask (Optional[torch.Tensor]): Self-attention mask of shape (batch_size, 1, seq_len, seq_len).
            cross_attention_mask (Optional[torch.Tensor]): Cross-attention mask of shape (batch_size, 1, seq_len, seq_len).
        
        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
        """
        _y: torch.Tensor = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)
        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)
        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Processes input through a sequence of decoder layers.
        
        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]):
                Encoder output, decoder input, self-attention mask, and cross-attention mask.
        
        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model).
        """
        x: torch.Tensor
        y: torch.Tensor
        self_attention_mask: Optional[torch.Tensor]
        cross_attention_mask: Optional[torch.Tensor]
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
        max_sequence_length: int,
        language_to_index: Dict[str, int],
        START_TOKEN: str,
        END_TOKEN: str,
        PADDING_TOKEN: str
    ):
        """
        Initializes the decoder module.
        Combines sentence embedding with multiple decoder layers.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            ffn_hidden (int): Dimension of the feed-forward hidden layer.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability.
            num_layers (int): Number of decoder layers.
            max_sequence_length (int): Maximum sequence length.
            language_to_index (Dict[str, int]): Mapping of tokens to indices.
            START_TOKEN (str): Start token.
            END_TOKEN (str): End token.
            PADDING_TOKEN (str): Padding token.
        """
        super().__init__()
        self.sentence_embedding: SentenceEmbedding = SentenceEmbedding(
            max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN
        )
        self.layers: SequentialDecoder = SequentialDecoder(
            *[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        y: List[str],
        self_attention_mask: Optional[torch.Tensor],
        cross_attention_mask: Optional[torch.Tensor],
        start_token: bool,
        end_token: bool
    ) -> torch.Tensor:
        """
        Processes input through the decoder.
        
        Args:
            x (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model).
            y (List[str]): Batch of target sentences.
            self_attention_mask (Optional[torch.Tensor]): Self-attention mask of shape (batch_size, 1, seq_len, seq_len).
            cross_attention_mask (Optional[torch.Tensor]): Cross-attention mask of shape (batch_size, 1, seq_len, seq_len).
            start_token (bool): Whether to prepend start token.
            end_token (bool): Whether to append end token.
        
        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, max_sequence_length, d_model).
        """
        y = self.sentence_embedding(y, start_token, end_token)
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
        kn_vocab_size: int,
        english_to_index: Dict[str, int],
        kannada_to_index: Dict[str, int],
        START_TOKEN: str,
        END_TOKEN: str,
        PADDING_TOKEN: str
    ):
        """
        Initializes the transformer model.
        Combines encoder, decoder, and final linear layer for translation.
        
        Args:
            d_model (int): Dimension of the model embeddings.
            ffn_hidden (int): Dimension of the feed-forward hidden layer.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability.
            num_layers (int): Number of encoder and decoder layers.
            max_sequence_length (int): Maximum sequence length.
            kn_vocab_size (int): Size of the target (Kannada) vocabulary.
            english_to_index (Dict[str, int]): Mapping of English tokens to indices.
            kannada_to_index (Dict[str, int]): Mapping of Kannada tokens to indices.
            START_TOKEN (str): Start token.
            END_TOKEN (str): End token.
            PADDING_TOKEN (str): Padding token.
        """
        super().__init__()
        self.encoder: Encoder = Encoder(
            d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length,
            english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN
        )
        self.decoder: Decoder = Decoder(
            d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length,
            kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN
        )
        self.linear: nn.Linear = nn.Linear(d_model, kn_vocab_size)
        self.device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(
        self,
        x: List[str],
        y: List[str],
        encoder_self_attention_mask: Optional[torch.Tensor] = None,
        decoder_self_attention_mask: Optional[torch.Tensor] = None,
        decoder_cross_attention_mask: Optional[torch.Tensor] = None,
        enc_start_token: bool = False,
        enc_end_token: bool = False,
        dec_start_token: bool = False,
        dec_end_token: bool = False
    ) -> torch.Tensor:
        """
        Processes input through the transformer model.
        
        Args:
            x (List[str]): Batch of source (English) sentences.
            y (List[str]): Batch of target (Kannada) sentences.
            encoder_self_attention_mask (Optional[torch.Tensor]): Encoder self-attention mask.
            decoder_self_attention_mask (Optional[torch.Tensor]): Decoder self-attention mask.
            decoder_cross_attention_mask (Optional[torch.Tensor]): Decoder cross-attention mask.
            enc_start_token (bool): Whether to prepend start token to encoder input.
            enc_end_token (bool): Whether to append end token to encoder input.
            dec_start_token (bool): Whether to prepend start token to decoder input.
            dec_end_token (bool): Whether to append end token to decoder input.
        
        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, max_sequence_length, kn_vocab_size).
        """
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out


        