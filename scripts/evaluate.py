import json
from pathlib import Path

import torch
from tqdm import tqdm

from config import load_config

from data.tokenizer import load_tokenizers

from evaluation.metrics import compute_bleu

from inference.translate import (
    load_model,
    translate_sentence,
)


config = load_config()

data_config = config["data"]
inference_config = config["inference"]
training_config = config["training"]


TEST_PATH = (
    Path(data_config["processed_dir"])
    / data_config["splits"]["test"]["output_file"]
)

CHECKPOINT_PATH = (
    Path(training_config["checkpoint_dir"])
    / inference_config["checkpoint_name"]
)

MAX_LEN = inference_config["max_len"]

BEAM_SIZE = inference_config["beam_size"]

LENGTH_PENALTY = inference_config[
    "length_penalty"
]

NUM_EXAMPLES = 10


def load_test_data(
    path: Path,
) -> list[tuple[str, str]]:
    """
    Load source-target pairs from a JSONL file.

    Args:
        path: Path to the test JSONL file.

    Returns:
        A list of (source, target) sentence pairs.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Test data not found: {path}"
        )

    samples = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            record = json.loads(line)

            if "src" not in record:
                raise KeyError(
                    "Missing 'src' field in test data."
                )

            if "tgt" not in record:
                raise KeyError(
                    "Missing 'tgt' field in test data."
                )

            samples.append(
                (
                    record["src"],
                    record["tgt"],
                )
            )

    if not samples:
        raise ValueError(
            f"No samples found in: {path}"
        )

    return samples


def generate_predictions(
    samples: list[tuple[str, str]],
    model,
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    beam_size: int,
) -> list[str]:
    """
    Generate translations for all test samples.

    Args:
        samples: Source-target sentence pairs.
        model: Trained Transformer model.
        src_tokenizer: Source tokenizer.
        tgt_tokenizer: Target tokenizer.
        device: Computation device.
        beam_size: Beam size. Use 1 for greedy decoding.

    Returns:
        Model-generated translations.
    """
    predictions = []

    decoding_name = (
        "Greedy"
        if beam_size == 1
        else f"Beam={beam_size}"
    )

    for src_text, _ in tqdm(
        samples,
        desc=f"Evaluating {decoding_name}",
    ):
        prediction = translate_sentence(
            model=model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            text=src_text,
            device=device,
            max_len=MAX_LEN,
            beam_size=beam_size,
            length_penalty=LENGTH_PENALTY,
        )

        predictions.append(
            prediction
        )

    return predictions


def print_qualitative_examples(
    samples: list[tuple[str, str]],
    predictions: list[str],
    num_examples: int,
) -> None:
    """
    Print source, reference, and prediction examples.

    Args:
        samples: Source-target sentence pairs.
        predictions: Model-generated translations.
        num_examples: Maximum number of examples to print.
    """
    print()
    print("=" * 70)
    print("QUALITATIVE EXAMPLES")
    print("=" * 70)

    num_examples = min(
        num_examples,
        len(samples),
    )

    for index in range(num_examples):
        src_text, tgt_text = samples[index]

        print()
        print(
            f"[Example {index + 1}]"
        )

        print(
            f"Source     : {src_text}"
        )

        print(
            f"Reference  : {tgt_text}"
        )

        print(
            f"Prediction : {predictions[index]}"
        )


def main() -> None:
    """Evaluate the translation model on the test set."""

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    en_tokenizer, vi_tokenizer = (
        load_tokenizers()
    )

    model = load_model(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        src_vocab_size=(
            en_tokenizer.vocab_size()
        ),
        tgt_vocab_size=(
            vi_tokenizer.vocab_size()
        ),
        src_pad_id=en_tokenizer.pad_id(),
        tgt_pad_id=vi_tokenizer.pad_id(),
    )

    samples = load_test_data(
        TEST_PATH
    )

    references = [
        tgt_text
        for _, tgt_text in samples
    ]

    # ------------------------------------------------------------------
    # Greedy decoding
    # ------------------------------------------------------------------

    greedy_predictions = generate_predictions(
        samples=samples,
        model=model,
        src_tokenizer=en_tokenizer,
        tgt_tokenizer=vi_tokenizer,
        device=device,
        beam_size=1,
    )

    greedy_bleu = compute_bleu(
        predictions=greedy_predictions,
        references=references,
    )

    # ------------------------------------------------------------------
    # Beam search
    # ------------------------------------------------------------------

    beam_predictions = generate_predictions(
        samples=samples,
        model=model,
        src_tokenizer=en_tokenizer,
        tgt_tokenizer=vi_tokenizer,
        device=device,
        beam_size=BEAM_SIZE,
    )

    beam_bleu = compute_bleu(
        predictions=beam_predictions,
        references=references,
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    print(
        f"Test samples : {len(samples):,}"
    )

    print(
        f"Greedy BLEU  : {greedy_bleu:.2f}"
    )

    print(
        f"Beam size    : {BEAM_SIZE}"
    )

    print(
        f"Beam BLEU    : {beam_bleu:.2f}"
    )

    print(
        f"Beam - Greedy: "
        f"{beam_bleu - greedy_bleu:+.2f}"
    )

    # ------------------------------------------------------------------
    # Qualitative examples
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("GREEDY QUALITATIVE EXAMPLES")
    print("=" * 70)

    print_qualitative_examples(
        samples=samples,
        predictions=greedy_predictions,
        num_examples=NUM_EXAMPLES,
    )

    print()
    print("=" * 70)
    print("BEAM SEARCH QUALITATIVE EXAMPLES")
    print("=" * 70)

    print_qualitative_examples(
        samples=samples,
        predictions=beam_predictions,
        num_examples=NUM_EXAMPLES,
    )


if __name__ == "__main__":
    main()