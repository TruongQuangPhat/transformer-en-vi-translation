import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from data.tokenizer import (
    SentencePieceTokenizer,
    pad_sequences,
)


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for English-Vietnamese translation.

    Each sample contains token IDs for the source and target sentences.
    """

    def __init__(
        self,
        jsonl_path: Path,
        src_tokenizer: SentencePieceTokenizer,
        tgt_tokenizer: SentencePieceTokenizer,
    ) -> None:
        """
        Args:
            jsonl_path: Path to the processed translation JSONL file.
            src_tokenizer: Tokenizer for the source language.
            tgt_tokenizer: Tokenizer for the target language.
        """
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {jsonl_path}"
            )

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.samples = self._load_data(jsonl_path)

    @staticmethod
    def _load_data(
        jsonl_path: Path,
    ) -> list[tuple[str, str]]:
        """
        Load source-target sentence pairs from a JSONL file.
        """
        samples = []

        with jsonl_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                record = json.loads(line)

                if "src" not in record or "tgt" not in record:
                    raise ValueError(
                        f"Missing 'src' or 'tgt' at line {line_number}."
                    )

                samples.append(
                    (record["src"], record["tgt"])
                )

        return samples

    def __len__(self) -> int:
        """Return the number of translation pairs."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """
        Return one tokenized translation sample.

        BOS and EOS are added to both source and target sequences.
        """
        src_text, tgt_text = self.samples[index]

        src_ids = self.src_tokenizer.encode_ids(
            src_text,
            add_bos=True,
            add_eos=True,
        )

        tgt_ids = self.tgt_tokenizer.encode_ids(
            tgt_text,
            add_bos=True,
            add_eos=True,
        )

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
        }

def collate_translation_batch(
    batch: list[dict[str, list[int]]],
    src_pad_id: int,
    tgt_pad_id: int,
) -> dict[str, torch.Tensor]:
    """
    Collate translation samples into padded tensors.

    Args:
        batch: List of tokenized translation samples.
        src_pad_id: Padding ID for the source language.
        tgt_pad_id: Padding ID for the target language.

    Returns:
        Dictionary containing padded source and target tensors.
    """
    src_sequences = [sample["src_ids"] for sample in batch]
    tgt_sequences = [sample["tgt_ids"] for sample in batch]

    src_padded = pad_sequences(
        sequences=src_sequences,
        pad_id=src_pad_id,
    )

    tgt_padded = pad_sequences(
        sequences=tgt_sequences,
        pad_id=tgt_pad_id,
    )

    return {
        "src_ids": torch.tensor(
            src_padded,
            dtype=torch.long,
        ),
        "tgt_ids": torch.tensor(
            tgt_padded,
            dtype=torch.long,
        ),
    }

def create_dataloader(
    dataset: TranslationDataset,
    batch_size: int,
    src_pad_id: int,
    tgt_pad_id: int,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for the translation dataset.

    Args:
        dataset: Translation dataset.
        batch_size: Number of samples per batch.
        src_pad_id: Padding ID for the source language.
        tgt_pad_id: Padding ID for the target language.
        shuffle: Whether to shuffle the dataset.

    Returns:
        Configured PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_translation_batch(
            batch=batch,
            src_pad_id=src_pad_id,
            tgt_pad_id=tgt_pad_id,
        ),
    )