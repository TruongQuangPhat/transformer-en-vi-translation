from torch import Tensor, nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward

class EncoderLayer(nn.Module):
    """
    A single Transformer Encoder layer.

    Architecture:
        Multi-Head Self-Attention
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

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout
    )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply one Encoder layer.

        Args:
            x: Input tensor with shape [B, S, d_model].

        Returns:
            A tuple containing:
                - output: [B, S, d_model]
                - attention weights: [B, num_heads, S, S]
        """
        # self-attention
        attention_output, attention_weights = self.self_attention(
            query=x,
            key=x,
            value=x,
        )
        x = self.norm1(x + self.dropout1(attention_output)) 
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(feed_forward_output))
        return x, attention_weights