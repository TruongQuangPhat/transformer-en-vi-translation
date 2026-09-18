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
    ) -> Tensor:
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