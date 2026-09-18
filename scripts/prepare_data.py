from pathlib import Path

from data.preprocessing import (
    inspect_dataset,
    load_parallel_data,
)


RAW_DATA_DIR = Path("data/raw")


def main() -> None:
    train_pairs = load_parallel_data(
        RAW_DATA_DIR / "train.en",
        RAW_DATA_DIR / "train.vi",
    )

    validation_pairs = load_parallel_data(
        RAW_DATA_DIR / "tst2012.en",
        RAW_DATA_DIR / "tst2012.vi",
    )

    test_pairs = load_parallel_data(
        RAW_DATA_DIR / "tst2013.en",
        RAW_DATA_DIR / "tst2013.vi",
    )

    inspect_dataset("train", train_pairs)
    inspect_dataset("validation", validation_pairs)
    inspect_dataset("test", test_pairs)


if __name__ == "__main__":
    main()