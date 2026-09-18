from torch import Tensor, nn

from model.attention import (
    MultiHeadAttention,
    combine_masks,
    create_causal_mask,
    create_padding_mask,
)
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

class Decoder(nn.Module):
    """
    Transformer Decoder composed of multiple Decoder layers.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            num_layers: Number of Decoder layers.
            d_model: Model representation dimension.
            num_heads: Number of attention heads.
            d_ff: Hidden dimension of the feed-forward network.
            dropout: Dropout probability.
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        target_ids: Tensor,
        source_ids: Tensor,
        tgt_pad_id: int,
        src_pad_id: int,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Apply all Decoder layers sequentially.

        Args:
            x: Decoder input with shape [B, S_tgt, d_model].
            memory: Encoder output with shape [B, S_src, d_model].
            target_ids: Target token IDs with shape [B, S_tgt].
            source_ids: Source token IDs with shape [B, S_src].
            tgt_pad_id: PAD ID for the target vocabulary.
            src_pad_id: PAD ID for the source vocabulary.

        Returns:
            A tuple containing:
                - output: [B, S_tgt, d_model]
                - self-attention weights from each layer:
                  [B, num_heads, S_tgt, S_tgt]
                - cross-attention weights from each layer:
                  [B, num_heads, S_tgt, S_src]
        """
        target_padding_mask = create_padding_mask(
            target_ids,
            pad_id=tgt_pad_id,
        )

        source_padding_mask = create_padding_mask(
            source_ids,
            pad_id=src_pad_id,
        )

        causal_mask = create_causal_mask(
            sequence_length=x.size(1),
            device=x.device,
        )

        self_attention_mask = combine_masks(
            target_padding_mask,
            causal_mask,
        )

        cross_attention_mask = source_padding_mask

        self_attention_weights = []
        cross_attention_weights = []

        for layer in self.layers:
            x, layer_self_weights, layer_cross_weights = layer(
                x=x,
                memory=memory,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
            )

            self_attention_weights.append(
                layer_self_weights
            )

            cross_attention_weights.append(
                layer_cross_weights
            )

        return (
            x,
            self_attention_weights,
            cross_attention_weights,
        )