from pathlib import Path
from statistics import mean, median
import re
import unicodedata
from html import unescape
import json

def load_parallel_data(
    src_path: Path,
    tgt_path: Path,
) -> list[tuple[str, str]]:
    """
    Load aligned source-target sentences from two text files.

    Each line in the source file is paired with the line at the
    same position in the target file.

    Args:
        src_path: Path to the source language file.
        tgt_path: Path to the target language file.

    Returns:
        A list of (source_sentence, target_sentence) pairs.

    Raises:
        FileNotFoundError:
            If either file does not exist.
        ValueError:
            If source and target files have different line counts.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    if not tgt_path.exists():
        raise FileNotFoundError(f"Target file not found: {tgt_path}")

    with (
        src_path.open("r", encoding="utf-8") as src_file,
        tgt_path.open("r", encoding="utf-8") as tgt_file,
    ):
        src_lines = [line.rstrip("\r\n") for line in src_file]
        tgt_lines = [line.rstrip("\r\n") for line in tgt_file]

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Alignment error:\n"
            f"  Source: {src_path} ({len(src_lines):,} lines)\n"
            f"  Target: {tgt_path} ({len(tgt_lines):,} lines)"
        )

    return list(zip(src_lines, tgt_lines))

def count_empty_pairs(
    pairs: list[tuple[str, str]],
) -> tuple[int, int]:
    """
    Count pairs with empty source or target sentences.

    Returns:
        A tuple of:
            - number of empty source sentences
            - number of empty target sentences
    """
    empty_src = sum(not src.strip() for src, _ in pairs)
    empty_tgt = sum(not tgt.strip() for _, tgt in pairs)

    return empty_src, empty_tgt

def count_duplicate_pairs(
    pairs: list[tuple[str, str]],
) -> int:
    """
    Count duplicate source-target sentence pairs.

    Returns:
        The number of duplicate pairs.
    """
    unique_pairs = set(pairs)
    return len(pairs) - len(unique_pairs)

def sentence_word_length(sentence: str) -> int:
    """
    Count whitespace-separated tokens in a sentence.

    This is only a rough inspection metric.
    Actual subword tokenization will be handled later.
    """
    return len(sentence.split())

def calculate_length_statistics(
    pairs: list[tuple[str, str]],
) -> dict[str, float]:
    """
    Calculate basic sentence length statistics.

    Returns:
        Statistics for source and target sentence word lengths.
    """
    src_lengths = [
        sentence_word_length(src)
        for src, _ in pairs
        if src.strip()
    ]

    tgt_lengths = [
        sentence_word_length(tgt)
        for _, tgt in pairs
        if tgt.strip()
    ]

    if not src_lengths or not tgt_lengths:
        return {
            "src_mean": 0.0,
            "src_median": 0.0,
            "src_min": 0,
            "src_max": 0,
            "tgt_mean": 0.0,
            "tgt_median": 0.0,
            "tgt_min": 0,
            "tgt_max": 0,
        }

    return {
        "src_mean": mean(src_lengths),
        "src_median": median(src_lengths),
        "src_min": min(src_lengths),
        "src_max": max(src_lengths),
        "tgt_mean": mean(tgt_lengths),
        "tgt_median": median(tgt_lengths),
        "tgt_min": min(tgt_lengths),
        "tgt_max": max(tgt_lengths),
    }


def find_extreme_pairs(
    pairs: list[tuple[str, str]],
    min_words: int = 2,
    max_words: int = 50,
    num_examples: int = 5,
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """
    Find very short and very long sentence pairs.

    Args:
        pairs: Sentence pairs to inspect.
        min_words: Maximum length considered very short.
        max_words: Minimum length considered very long.
        num_examples: Number of examples to return.

    Returns:
        A tuple containing:
            - short sentence examples
            - long sentence examples
    """
    short_pairs = [
        pair
        for pair in pairs
        if sentence_word_length(pair[0]) <= min_words
        or sentence_word_length(pair[1]) <= min_words
    ]

    long_pairs = [
        pair
        for pair in pairs
        if sentence_word_length(pair[0]) >= max_words
        or sentence_word_length(pair[1]) >= max_words
    ]

    return short_pairs[:num_examples], long_pairs[:num_examples]


def inspect_dataset(
    split_name: str,
    pairs: list[tuple[str, str]],
) -> None:
    """
    Print inspection statistics for one dataset split.
    """
    empty_src, empty_tgt = count_empty_pairs(pairs)
    duplicate_pairs = count_duplicate_pairs(pairs)
    length_stats = calculate_length_statistics(pairs)

    short_pairs, long_pairs = find_extreme_pairs(pairs)

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} DATASET")
    print("=" * 70)

    print(f"Sentence pairs : {len(pairs):,}")
    print(f"Empty source   : {empty_src:,}")
    print(f"Empty target   : {empty_tgt:,}")
    print(f"Duplicate pairs: {duplicate_pairs:,}")

    print("\nSentence length statistics (whitespace tokens):")

    print(
        f"  EN → mean={length_stats['src_mean']:.2f}, "
        f"median={length_stats['src_median']:.2f}, "
        f"min={length_stats['src_min']}, "
        f"max={length_stats['src_max']}"
    )

    print(
        f"  VI → mean={length_stats['tgt_mean']:.2f}, "
        f"median={length_stats['tgt_median']:.2f}, "
        f"min={length_stats['tgt_min']}, "
        f"max={length_stats['tgt_max']}"
    )

    if short_pairs:
        print("\nVery short examples:")

        for index, (src, tgt) in enumerate(short_pairs, start=1):
            print(f"\n  [{index}] EN: {src}")
            print(f"      VI: {tgt}")

    if long_pairs:
        print("\nVery long examples:")

        for index, (src, tgt) in enumerate(long_pairs, start=1):
            print(f"\n  [{index}] EN: {src}")
            print(f"      VI: {tgt}")

def normalize_sentence(sentence: str) -> str:
    """
    Normalize text without changing its linguistic content.

    Operations:
        1. Unicode normalization (NFC)
        2. HTML entity decoding
        3. Collapse repeated whitespace
        4. Strip leading/trailing whitespace
    """
    sentence = unicodedata.normalize("NFC", sentence)
    sentence = unescape(sentence)
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()

def clean_parallel_data(
    pairs: list[tuple[str, str]],
    max_words: int = 128,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """
    Clean and filter parallel sentence pairs.

    Returns:
        cleaned_pairs:
            Cleaned source-target pairs.

        stats:
            Statistics describing how many pairs were removed.
    """
    original_count = len(pairs)

    normalized_pairs: list[tuple[str, str]] = []

    empty_pairs_removed = 0
    overlong_pairs_removed = 0

    for src, tgt in pairs:
        src = normalize_sentence(src)
        tgt = normalize_sentence(tgt)

        if not src or not tgt:
            empty_pairs_removed += 1
            continue

        if (
            sentence_word_length(src) > max_words
            or sentence_word_length(tgt) > max_words
        ):
            overlong_pairs_removed += 1
            continue

        normalized_pairs.append((src, tgt))

    unique_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for pair in normalized_pairs:
        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)
        unique_pairs.append(pair)

    duplicate_pairs_removed = (
        len(normalized_pairs) - len(unique_pairs)
    )

    stats = {
        "original_pairs": original_count,
        "empty_pairs_removed": empty_pairs_removed,
        "overlong_pairs_removed": overlong_pairs_removed,
        "duplicate_pairs_removed": duplicate_pairs_removed,
        "final_pairs": len(unique_pairs),
        "total_removed": original_count - len(unique_pairs),
    }

    return unique_pairs, stats

def save_parallel_data(
    pairs: list[tuple[str, str]],
    output_path: Path,
) -> None:
    """
    Save parallel sentence pairs as JSONL.

    Each line contains one JSON object with:
        - src: source sentence
        - tgt: target sentence
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for src, tgt in pairs:
            record = {
                "src": src,
                "tgt": tgt,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )