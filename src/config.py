from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict:
    """
    Load YAML configuration from a file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid config format: {config_path}"
        )

    return config