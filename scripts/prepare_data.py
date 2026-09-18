from pathlib import Path

from data.preprocessing import load_parallel_data


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

    print(f"Train pairs: {len(train_pairs):,}")
    print(f"Validation pairs: {len(validation_pairs):,}")
    print(f"Test pairs: {len(test_pairs):,}")

    print("\nFirst 3 training examples:")

    for index, (src, tgt) in enumerate(train_pairs[:3], start=1):
        print(f"\nExample {index}")
        print(f"EN: {src}")
        print(f"VI: {tgt}")


if __name__ == "__main__":
    main()