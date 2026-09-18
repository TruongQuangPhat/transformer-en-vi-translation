from pathlib import Path

from data.preprocessing import (
    clean_parallel_data,
    inspect_dataset,
    load_parallel_data,
)


RAW_DATA_DIR = Path("data/raw")
MAX_WORDS = 128


def print_raw_summary(
    pairs: list[tuple[str, str]],
) -> None:
    """Print a compact summary of the raw dataset."""
    empty_src, empty_tgt = 0, 0

    for src, tgt in pairs:
        if not src.strip():
            empty_src += 1

        if not tgt.strip():
            empty_tgt += 1

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
) -> None:
    """Load and clean one dataset split."""

    raw_pairs = load_parallel_data(
        RAW_DATA_DIR / src_filename,
        RAW_DATA_DIR / tgt_filename,
    )

    cleaned_pairs, stats = clean_parallel_data(
        raw_pairs,
        max_words=MAX_WORDS,
    )

    print("\n" + "=" * 70)
    print(f"{split_name.upper()}")
    print("=" * 70)

    print("\n[RAW]")
    print_raw_summary(raw_pairs)

    print("\n[CLEANING]")
    print_cleaning_summary(stats)

    print("\n[AFTER CLEANING]")
    empty_src, empty_tgt = 0, 0

    for src, tgt in cleaned_pairs:
        if not src.strip():
            empty_src += 1

        if not tgt.strip():
            empty_tgt += 1

    print(f"Pairs        : {len(cleaned_pairs):,}")
    print(f"Empty source : {empty_src:,}")
    print(f"Empty target : {empty_tgt:,}")

    print("\n[EXAMPLES]")

    for index, (src, tgt) in enumerate(cleaned_pairs[:3], start=1):
        print(f"{index}. EN: {src}")
        print(f"   VI: {tgt}")


def main() -> None:
    process_split(
        "train",
        "train.en",
        "train.vi",
    )

    process_split(
        "validation",
        "tst2012.en",
        "tst2012.vi",
    )

    process_split(
        "test",
        "tst2013.en",
        "tst2013.vi",
    )


if __name__ == "__main__":
    main()