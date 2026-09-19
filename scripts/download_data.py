from pathlib import Path
from urllib.request import Request, urlopen

from config import load_config


config = load_config()

data_config = config["data"]
download_config = data_config["download"]

DATA_DIR = Path(
    data_config["raw_dir"]
)

BASE_URL = download_config["base_url"]

FILES = download_config["files"]


def download_file(
    filename: str,
) -> Path:
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

    with urlopen(
        request,
        timeout=120,
    ) as response:
        data = response.read()

    output_path.write_bytes(data)

    size_mb = len(data) / (1024 * 1024)

    print(
        f"[OK] {filename}: "
        f"{size_mb:.2f} MB"
    )

    return output_path


def main() -> None:
    """Download all configured dataset archives."""
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split, filename in FILES.items():
        print(
            f"\nProcessing {split} split"
        )

        download_file(filename)


if __name__ == "__main__":
    main()