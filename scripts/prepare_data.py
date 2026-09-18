from pathlib import Path

from data.preprocessing import (
    clean_parallel_data,
    load_parallel_data,
    save_parallel_data,
)


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

MAX_WORDS = 128


def print_raw_summary(
    pairs: list[tuple[str, str]],
) -> None:
    """Print a compact summary of the raw dataset."""
    empty_src = sum(not src.strip() for src, _ in pairs)
    empty_tgt = sum(not tgt.strip() for _, tgt in pairs)

    print(f"Pairs        : {len(pairs):,}")
    print(f"Empty source : {empty_src:,}")
    print(f"Empty target : {empty_tgt:,}")


def print_cleaning_summary(
    stats: dict[str, int],
) -> None:
    """Print a compact summary of cleaning operations."""
    print(f"Original pairs          : {stats['original_pairs']:,}")
    print(f"Empty pairs removed     : {stats['empty_pairs_removed']:,}")
    print(f"Overlong pairs removed  : {stats['overlong_pairs_removed']:,}")
    print(f"Duplicate pairs removed : {stats['duplicate_pairs_removed']:,}")
    print(f"Total removed           : {stats['total_removed']:,}")
    print(f"Final pairs             : {stats['final_pairs']:,}")


def process_split(
    split_name: str,
    src_filename: str,
    tgt_filename: str,
    output_filename: str,
) -> None:
    """
    Load, clean, summarize, and save one dataset split.
    """
    raw_pairs = load_parallel_data(
        RAW_DATA_DIR / src_filename,
        RAW_DATA_DIR / tgt_filename,
    )

    cleaned_pairs, stats = clean_parallel_data(
        raw_pairs,
        max_words=MAX_WORDS,
    )

    output_path = PROCESSED_DATA_DIR / output_filename

    save_parallel_data(
        cleaned_pairs,
        output_path,
    )

    print("\n" + "=" * 70)
    print(f"{split_name.upper()}")
    print("=" * 70)

    print("\n[RAW]")
    print_raw_summary(raw_pairs)

    print("\n[CLEANING]")
    print_cleaning_summary(stats)

    print("\n[SAVED]")
    print(f"Path : {output_path}")
    print(f"Pairs: {len(cleaned_pairs):,}")


def main() -> None:
    process_split(
        split_name="train",
        src_filename="train.en",
        tgt_filename="train.vi",
        output_filename="train.jsonl",
    )

    process_split(
        split_name="validation",
        src_filename="tst2012.en",
        tgt_filename="tst2012.vi",
        output_filename="validation.jsonl",
    )

    process_split(
        split_name="test",
        src_filename="tst2013.en",
        tgt_filename="tst2013.vi",
        output_filename="test.jsonl",
    )


if __name__ == "__main__":
    main()