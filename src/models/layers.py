import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input sequence embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # --- Pre-calculate Positional Encodings ---
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Handle odd d_model case
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
             pe[:, 1::2] = torch.cos(position * div_term[:-1])

        # Shape [1, max_len, d_model] for batch_first=True
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        """
        # Slice to the current sequence length
        x = x + self.pe[:, :x.size(1), :] 
        return self.dropout(x)