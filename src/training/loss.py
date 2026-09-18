import torch
from torch import Tensor, nn

class TranslationLoss(nn.Module):
    """Cross-entropy loss for sequence-to-sequence translation."""
    def __init__(self, pad_id: int) -> None:
        super().__init__()

        self.loss = nn.CrossEntropyLoss(
            ignore_index=pad_id,
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model outputs with shape [B, S, V].
            targets: Target IDs with shape [B, S].

        Returns:
            Scalar loss.
        """
        batch_size, sequence_length, vocab_size = logits.shape

        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size,
        )

        targets = targets.reshape(
            batch_size * sequence_length,
        )

        return self.loss(logits, targets)