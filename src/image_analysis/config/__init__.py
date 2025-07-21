import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_segmenter_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load segmenter configuration from YAML file.

    Args:
        config_path: Path to the configuration file. If None, uses default config.

    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        # Use default config file in the same directory as this module
        config_path = str(Path(__file__).parent / "segmenter_config.yaml")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_segmenter_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save segmenter configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
