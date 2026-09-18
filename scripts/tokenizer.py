import argparse

from data.tokenizer import load_tokenizers, train_all_tokenizers


def test_tokenizers() -> None:
    """
    Demonstrate tokenization and decoding for English and Vietnamese.
    """
    en_tokenizer, vi_tokenizer = load_tokenizers()

    examples = [
        (
            "English",
            en_tokenizer,
            "I am learning machine translation.",
        ),
        (
            "Vietnamese",
            vi_tokenizer,
            "Tôi đang học máy dịch.",
        ),
    ]

    for language, tokenizer, text in examples:
        pieces = tokenizer.encode_pieces(text)
        ids = tokenizer.encode_ids(text)
        decoded = tokenizer.decode(ids)

        print(f"\n[{language}]")
        print(f"Text     : {text}")
        print(f"Pieces   : {pieces}")
        print(f"IDs      : {ids}")
        print(f"Decoded  : {decoded}")
        print(f"Vocab    : {tokenizer.vocab_size()}")

        if decoded != text:
            print("Warning  : decoded text differs from original text.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and test English-Vietnamese tokenizers."
    )

    parser.add_argument(
        "command",
        choices={"train", "test"},
        help="Choose whether to train or test the tokenizers.",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_all_tokenizers()
        print("\nTokenizer training completed.")

    elif args.command == "test":
        test_tokenizers()


if __name__ == "__main__":
    main()