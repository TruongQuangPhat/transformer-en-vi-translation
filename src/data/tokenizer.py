import json
from pathlib import Path

import sentencepiece as spm


TRAIN_DATA_PATH = Path("data/processed/train.jsonl")
TOKENIZER_DIR = Path("data/tokenizer")

VOCAB_SIZE = 8_000

UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def prepare_tokenizer_corpus(
    jsonl_path: Path,
    output_path: Path,
    language: str,
) -> None:
    """
    Extract one language from a translation JSONL file.

    Each output line contains one sentence for SentencePiece training.

    Args:
        jsonl_path: Path to the processed translation dataset.
        output_path: Path to the extracted text corpus.
        language: Either "src" or "tgt".
    """
    if language not in {"src", "tgt"}:
        raise ValueError("language must be 'src' or 'tgt'")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        jsonl_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            record = json.loads(line)
            output_file.write(record[language] + "\n")


def train_tokenizer(
    corpus_path: Path,
    model_prefix: Path,
) -> None:
    """
    Train a SentencePiece BPE tokenizer.

    Args:
        corpus_path: Path to the training text corpus.
        model_prefix: Output prefix for the SentencePiece model files.
    """
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )


def train_all_tokenizers() -> None:
    """
    Prepare training corpora and train both language tokenizers.
    """
    en_corpus = TOKENIZER_DIR / "train.en.txt"
    vi_corpus = TOKENIZER_DIR / "train.vi.txt"

    en_model_prefix = TOKENIZER_DIR / "tokenizer_en"
    vi_model_prefix = TOKENIZER_DIR / "tokenizer_vi"

    prepare_tokenizer_corpus(
        jsonl_path=TRAIN_DATA_PATH,
        output_path=en_corpus,
        language="src",
    )

    prepare_tokenizer_corpus(
        jsonl_path=TRAIN_DATA_PATH,
        output_path=vi_corpus,
        language="tgt",
    )

    train_tokenizer(
        corpus_path=en_corpus,
        model_prefix=en_model_prefix,
    )

    train_tokenizer(
        corpus_path=vi_corpus,
        model_prefix=vi_model_prefix,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class SentencePieceTokenizer:
    """Wrapper around a trained SentencePiece tokenizer."""

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}"
            )

        self.processor = spm.SentencePieceProcessor(
            model_file=str(model_path)
        )

    def encode_pieces(self, text: str) -> list[str]:
        """
        Convert text into SentencePiece subword pieces.
        """
        return self.processor.encode(
            text,
            out_type=str,
        )

    def encode_ids(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Convert text into token IDs.

        Args:
            text: Input sentence.
            add_bos: Whether to prepend the BOS token.
            add_eos: Whether to append the EOS token.
        """
        ids = self.processor.encode(
            text,
            out_type=int,
        )

        if add_bos:
            ids.insert(0, self.bos_id())

        if add_eos:
            ids.append(self.eos_id())

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Convert token IDs back into text.
        """
        return self.processor.decode(ids)

    def vocab_size(self) -> int:
        """
        Return the vocabulary size.
        """
        return self.processor.get_piece_size()

    def unk_id(self) -> int:
        """Return the UNK token ID."""
        return self.processor.unk_id()

    def bos_id(self) -> int:
        """Return the BOS token ID."""
        return self.processor.bos_id()

    def eos_id(self) -> int:
        """Return the EOS token ID."""
        return self.processor.eos_id()

    def pad_id(self) -> int:
        """Return the PAD token ID."""
        return self.processor.pad_id()

def pad_sequences(
    sequences: list[list[int]],
    pad_id: int,
    max_length: int | None = None,
) -> list[list[int]]:
    """
    Pad sequences to the same length.

    Args:
        sequences: Token ID sequences.
        pad_id: ID used for padding.
        max_length: Target length. If None, use the longest sequence.

    Returns:
        Padded sequences.
    """
    if not sequences:
        return []

    if max_length is None:
        max_length = max(len(sequence) for sequence in sequences)

    padded_sequences = []

    for sequence in sequences:
        if len(sequence) > max_length:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds max_length "
                f"{max_length}."
            )

        padded_sequence = sequence + [pad_id] * (
            max_length - len(sequence)
        )

        padded_sequences.append(padded_sequence)

    return padded_sequences


def load_tokenizers() -> tuple[
    SentencePieceTokenizer,
    SentencePieceTokenizer,
]:
    """
    Load the trained English and Vietnamese tokenizers.

    Returns:
        A tuple containing:
            - English tokenizer
            - Vietnamese tokenizer
    """
    en_tokenizer = SentencePieceTokenizer(
        TOKENIZER_DIR / "tokenizer_en.model"
    )

    vi_tokenizer = SentencePieceTokenizer(
        TOKENIZER_DIR / "tokenizer_vi.model"
    )

    return en_tokenizer, vi_tokenizer