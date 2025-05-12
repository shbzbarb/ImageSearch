import yaml
from typing import Dict, Any
import os

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))

def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any] | None:
    """
    Loads the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration, or None if loading fails.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        return None