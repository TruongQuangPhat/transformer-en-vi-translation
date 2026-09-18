import math

import torch
from torch import Tensor, nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Add sinusoidal positional information to token embeddings.
    """
    def __init__(
        self,
        d_model: int,
        max_len: int = 512
    ) -> None:
        """
        Args:
            d_model: Embedding dimension.
            max_len: Maximum supported sequence length.
        """
        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input embeddings.

        Args:
            x: Tensor with shape [B, S, d_model].

        Returns:
            Tensor with shape [B, S, d_model].
        """
        sequence_length = x.size(1)

        if sequence_length > self.pe.size(1):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum length {self.pe.size(1)}."
            )

        return x + self.pe[:, :sequence_length]