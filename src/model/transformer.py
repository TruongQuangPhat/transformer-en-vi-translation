from torch import Tensor, nn

from model.attention import create_padding_mask
from model.decoder import Decoder
from model.embeddings import TokenEmbedding
from model.encoder import Encoder
from model.positional_encoding import SinusoidalPositionalEncoding

class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.

    Architecture:
        Encoder
        -> Decoder
        -> Linear + Softmax
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            src_vocab_size: Source vocabulary size.
            tgt_vocab_size: Target vocabulary size.
            src_pad_id: PAD ID of the source vocabulary.
            tgt_pad_id: PAD ID of the target vocabulary.
            d_model: Model representation dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            num_encoder_layers: Number of Encoder layers.
            num_decoder_layers: Number of Decoder layers.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()

        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

        self.src_embedding = TokenEmbedding(
            vocab_size=src_vocab_size,
            d_model=d_model,
            pad_id=src_pad_id,
        )

        self.tgt_embedding = TokenEmbedding(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            pad_id=tgt_pad_id,
        )

        self.src_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        self.tgt_positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size,
        )

    def forward(
        self,
        src_ids: Tensor,
        tgt_ids: Tensor,
    ) -> tuple[
        Tensor,
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        """
        Run the complete Transformer.

        Args:
            src_ids: Source token IDs with shape [B, S_src].
            tgt_ids: Target token IDs with shape [B, S_tgt].

        Returns:
            A tuple containing:
                - logits: [B, S_tgt, tgt_vocab_size]
                - encoder attention weights
                - decoder self-attention weights
                - decoder cross-attention weights
        """
        # Source embedding + positional encoding
        src = self.src_embedding(src_ids)
        src = self.src_positional_encoding(src)

        # Source padding mask for Encoder self-attention
        src_padding_mask = create_padding_mask(
            src_ids,
            pad_id=self.src_pad_id,
        )

        # Encoder
        memory, encoder_attention_weights = self.encoder(
            x=src,
            self_attention_mask=src_padding_mask,
        )

        # Target embedding + positional encoding
        tgt = self.tgt_embedding(tgt_ids)
        tgt = self.tgt_positional_encoding(tgt)

        # Decoder
        decoder_output, decoder_self_attention_weights, decoder_cross_attention_weights = self.decoder(
            x=tgt,
            memory=memory,
            target_ids=tgt_ids,
            source_ids=src_ids,
            tgt_pad_id=self.tgt_pad_id,
            src_pad_id=self.src_pad_id,
        )

        # Project decoder representations to target vocabulary logits
        logits = self.output_projection(
            decoder_output
        )

        return (
            logits,
            encoder_attention_weights,
            decoder_self_attention_weights,
            decoder_cross_attention_weights,
        )