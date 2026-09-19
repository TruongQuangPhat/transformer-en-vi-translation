from pathlib import Path

from config import load_config

from data.preprocessing import (
    clean_parallel_data,
    load_parallel_data,
    save_parallel_data,
)


config = load_config()

data_config = config["data"]

RAW_DATA_DIR = Path(
    data_config["raw_dir"]
)

PROCESSED_DATA_DIR = Path(
    data_config["processed_dir"]
)

MAX_WORDS = data_config[
    "preprocessing"
]["max_words"]

SPLITS = data_config["splits"]


def print_raw_summary(
    pairs: list[tuple[str, str]],
) -> None:
    """Print a compact summary of the raw dataset."""
    empty_src = sum(
        not src.strip()
        for src, _ in pairs
    )

    empty_tgt = sum(
        not tgt.strip()
        for _, tgt in pairs
    )

    print(
        f"Pairs        : {len(pairs):,}"
    )

    print(
        f"Empty source : {empty_src:,}"
    )

    print(
        f"Empty target : {empty_tgt:,}"
    )


def print_cleaning_summary(
    stats: dict[str, int],
) -> None:
    """Print a compact summary of cleaning operations."""
    print(
        f"Original pairs          : "
        f"{stats['original_pairs']:,}"
    )

    print(
        f"Empty pairs removed     : "
        f"{stats['empty_pairs_removed']:,}"
    )

    print(
        f"Overlong pairs removed  : "
        f"{stats['overlong_pairs_removed']:,}"
    )

    print(
        f"Duplicate pairs removed : "
        f"{stats['duplicate_pairs_removed']:,}"
    )

    print(
        f"Total removed           : "
        f"{stats['total_removed']:,}"
    )

    print(
        f"Final pairs             : "
        f"{stats['final_pairs']:,}"
    )


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

    cleaned_pairs, stats = (
        clean_parallel_data(
            raw_pairs,
            max_words=MAX_WORDS,
        )
    )

    output_path = (
        PROCESSED_DATA_DIR
        / output_filename
    )

    save_parallel_data(
        cleaned_pairs,
        output_path,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        split_name.upper()
    )

    print(
        "=" * 70
    )

    print("\n[RAW]")

    print_raw_summary(
        raw_pairs
    )

    print("\n[CLEANING]")

    print_cleaning_summary(
        stats
    )

    print("\n[SAVED]")

    print(
        f"Path : {output_path}"
    )

    print(
        f"Pairs: {len(cleaned_pairs):,}"
    )


def main() -> None:
    """Process all configured dataset splits."""
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name, split_config in SPLITS.items():
        process_split(
            split_name=split_name,
            src_filename=split_config[
                "src_file"
            ],
            tgt_filename=split_config[
                "tgt_file"
            ],
            output_filename=split_config[
                "output_file"
            ],
        )


if __name__ == "__main__":
    main()