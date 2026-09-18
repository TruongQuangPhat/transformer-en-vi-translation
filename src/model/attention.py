import math

import torch
from torch import Tensor, nn

class ScaledDotProductAttention(nn.Module):
    """
    Compute scaled dot-product attention.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the attention output.

        Args:
            query: Query tensor with shape [B, S_q, d_k].
            key: Key tensor with shape [B, S_k, d_k].
            value: Value tensor with shape [B, S_v, d_v].
        Returns:
            Attention output tensor with shape [B, S_q, d_v].
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Compute multi-head attention.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ) -> None:
        """
        Args:
            d_model: Model representation dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # d_k

        # Q
        self.q_proj = nn.Linear(
            d_model,
            d_model,
        )
        # K
        self.k_proj = nn.Linear(
            d_model,
            d_model,
        )
        # V
        self.v_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.attention = ScaledDotProductAttention()

        self.out_proj = nn.Linear(
            d_model,
            d_model,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Split the model dimension into multiple attention heads.

        Args:
            x: Tensor with shape [B, S, d_model].

        Returns:
            Tensor with shape [B, num_heads, S, head_dim].
        """
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Combine multiple attention heads.

        Args:
            x: Tensor with shape [B, num_heads, S, head_dim].

        Returns:
            Tensor with shape [B, S, d_model].
        """
        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            sequence_length,
            self.d_model,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor with shape [B, S_q, d_model].
            key: Key tensor with shape [B, S_k, d_model].
            value: Value tensor with shape [B, S_v, d_model].

        Returns:
        A tuple containing:
            - Attention output tensor with shape [B, S_q, d_model].
            - Attention weights with shape
            [B, num_heads, S_q, S_k].
        """
        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Split into heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Compute attention
        attention_output, attention_weights = self.attention(Q, K, V)

        # Combine heads
        combined_output = self._combine_heads(attention_output)

        # Final linear projection
        output = self.out_proj(combined_output)

        return output, attention_weights