import argparse

from data.tokenizer import (
    load_tokenizers,
    pad_sequences,
    train_all_tokenizers,
)


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
        ids = tokenizer.encode_ids(text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids)

        print(f"\n[{language}]")
        print(f"Text       : {text}")
        print(f"Pieces     : {pieces}")
        print(f"IDs        : {ids}")
        print(f"Decoded    : {decoded}")
        print(f"Vocab size : {tokenizer.vocab_size()}")

        print("\nSpecial tokens:")
        print(f"UNK = {tokenizer.unk_id()}")
        print(f"BOS = {tokenizer.bos_id()}")
        print(f"EOS = {tokenizer.eos_id()}")
        print(f"PAD = {tokenizer.pad_id()}")

        if decoded != text:
            print("Warning    : decoded text differs from original text.")

    sequences = [
        en_tokenizer.encode_ids("I am learning."),
        en_tokenizer.encode_ids("I love machine learning."),
        en_tokenizer.encode_ids("Hello."),
    ]

    padded_sequences = pad_sequences(
        sequences,
        pad_id=en_tokenizer.pad_id(),
    )

    print("\nPadding example:")
    for sequence, padded in zip(sequences, padded_sequences):
        print(f"{sequence} -> {padded}")


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