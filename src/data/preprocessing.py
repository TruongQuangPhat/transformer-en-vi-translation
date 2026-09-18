from pathlib import Path


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