from torch import Tensor, nn

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            d_model: Model representation dimension.
            d_ff: Hidden layer dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(
            d_ff,
            d_model,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the feed-forward network.

        Args:
            x: Tensor with shape [B, S, d_model].

        Returns:
            Tensor with shape [B, S, d_model].
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

        
    