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
    Prepare corpora and train English and Vietnamese tokenizers.
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


if __name__ == "__main__":
    train_all_tokenizers()