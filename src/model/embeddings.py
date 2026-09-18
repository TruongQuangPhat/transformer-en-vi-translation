import math

import torch
from torch import Tensor, nn

class TokenEmbedding(nn.Module):
    """
    Convert token IDs into dense embedding vectors.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: int,
    ) -> None:
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            d_model: Embedding dimension.
            pad_id: ID of the PAD token.
        """
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Convert token IDs into embedding vectors.

        Args:
            token_ids: Tensor with shape [B, S].

        Returns:
            Tensor with shape [B, S, d_model].
        """
        embeddings = self.embedding(token_ids)
        embeddings *= math.sqrt(self.d_model)
        return embeddings
