import re
import sacrebleu


def detokenize_text(text: str) -> str:
    """Normalize common tokenization artifacts for evaluation."""
    text = re.sub(r"\s+([,.!?;:%])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_bleu(
    predictions: list[str],
    references: list[str],
) -> float:
    """Compute corpus-level BLEU after detokenization."""
    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must "
            "contain the same number of sentences."
        )

    predictions = [
        detokenize_text(text)
        for text in predictions
    ]

    references = [
        detokenize_text(text)
        for text in references
    ]

    result = sacrebleu.corpus_bleu(
        predictions,
        [references],
    )

    return result.score