from pathlib import Path
from urllib.request import Request, urlopen


DATA_DIR = Path("data/raw")

BASE_URL = "https://github.com/stefan-it/nmt-en-vi/raw/master/data"

FILES = {
    "train": "train-en-vi.tgz",
    "validation": "dev-2012-en-vi.tgz",
    "test": "test-2013-en-vi.tgz",
}


def download_file(filename: str) -> Path:
    """Download one dataset archive."""
    output_path = DATA_DIR / filename

    if output_path.exists():
        print(f"[SKIP] {output_path}")
        return output_path

    url = f"{BASE_URL}/{filename}"
    print(f"[DOWNLOAD] {url}")

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urlopen(request, timeout=120) as response:
        data = response.read()

    output_path.write_bytes(data)

    size_mb = len(data) / (1024 * 1024)
    print(f"[OK] {filename}: {size_mb:.2f} MB")

    return output_path


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, filename in FILES.items():
        print(f"\nProcessing {split} split")
        download_file(filename)


if __name__ == "__main__":
    main()