import json
from pathlib import Path

from torch.utils.data import Dataset

from data.tokenizer import SentencePieceTokenizer


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