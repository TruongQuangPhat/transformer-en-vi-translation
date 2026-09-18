import math

import torch
from torch import Tensor, nn

def create_padding_mask(
    token_ids: Tensor,
    pad_id: int,
) -> Tensor:
    """
    Create a padding mask.

    Args:
        token_ids: Token IDs with shape [B, S].
        pad_id: ID of the PAD token.

    Returns:
        Boolean mask with shape [B, 1, 1, S].
        True indicates a padded position.
    """
    return (token_ids == pad_id).unsqueeze(1).unsqueeze(2)

def create_causal_mask(
    sequence_length: int,
    device: torch.device | None = None,
) -> Tensor:
    """
    Create a causal attention mask.

    Args:
        sequence_length: Target sequence length.
        device: Device for the mask.

    Returns:
        Boolean mask with shape [1, 1, S, S].
        True indicates a future position that must be masked.
    """
    mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )

    return mask.unsqueeze(0).unsqueeze(0)

def combine_masks(
    *masks: Tensor | None,
) -> Tensor | None:
    """
    Combine multiple attention masks using logical OR.

    Args:
        masks: Attention masks. None values are ignored.

    Returns:
        Combined attention mask, or None if no mask is provided.
    """
    valid_masks = [
        mask
        for mask in masks
        if mask is not None
    ]

    if not valid_masks:
        return None

    combined_mask = valid_masks[0]

    for mask in valid_masks[1:]:
        combined_mask = combined_mask | mask

    return combined_mask

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
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the attention output.

        Args:
            query: Query tensor with shape [..., S_q, d_k].
            key: Key tensor with shape [..., S_k, d_k].
            value: Value tensor with shape [..., S_k, d_v].
            mask: Boolean attention mask. True means the position
                  should be masked.
        Returns:
            A tuple containing:
            - Attention output with shape [..., S_q, d_v].
            - Attention weights with shape [..., S_q, S_k].
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask,
                float("-inf"),
            )

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
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor with shape [B, S_q, d_model].
            key: Key tensor with shape [B, S_k, d_model].
            value: Value tensor with shape [B, S_v, d_model].
            mask: Boolean attention mask. True means the position
                  should be masked.

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
        attention_output, attention_weights = self.attention(Q, K, V, mask)

        # Combine heads
        combined_output = self._combine_heads(attention_output)

        # Final linear projection
        output = self.out_proj(combined_output)

        return output, attention_weights