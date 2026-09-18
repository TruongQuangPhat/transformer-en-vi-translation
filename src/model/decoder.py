from torch import Tensor, nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward

class DecoderLayer(nn.Module):
    """
    A single Transformer Decoder layer.

    Architecture:
        Masked Self-Attention
        -> Residual + LayerNorm
        -> Cross-Attention
        -> Residual + LayerNorm
        -> Feed Forward
        -> Residual + LayerNorm
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            d_model: Model representation dimension.
            num_heads: Number of attention heads.
            d_ff: Hidden dimension of the feed-forward network.
            dropout: Dropout probability.
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Apply one Decoder layer.

        Args:
            x: Input tensor with shape [B, S_tgt, d_model].
            memory: Encoder output tensor with shape [B, S_src, d_model].
            self_attention_mask: Boolean mask for self-attention.
            cross_attention_mask: Boolean mask for cross-attention.

        Returns:
            A tuple containing:
                - output: [B, S_tgt, d_model]
                - self-attention weights: [B, num_heads, S_tgt, S_tgt]
                - cross-attention weights: [B, num_heads, S_tgt, S_src]
        """
        # 1. Masked self-attention
        self_attention_output, self_attention_weights = (
            self.self_attention(
                query=x,
                key=x,
                value=x,
                mask=self_attention_mask,
            )
        )

        x = self.norm1(
            x + self.dropout1(self_attention_output)
        )

        # 2. Cross-attention
        cross_attention_output, cross_attention_weights = (
            self.cross_attention(
                query=x,
                key=memory, 
                value=memory,
                mask=cross_attention_mask,
            )
        )

        x = self.norm2(
            x + self.dropout2(cross_attention_output)
        )

        # 3. Feed-forward network
        feed_forward_output = self.feed_forward(x)

        x = self.norm3(
            x + self.dropout3(feed_forward_output)
        )

        return (
            x,
            self_attention_weights,
            cross_attention_weights,
        )

    