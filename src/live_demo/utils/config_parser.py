"""Configuration parser for live demo YAML config files."""
import yaml
from pathlib import Path
from typing import Dict, Any


def parse_config(config_path: str) -> Dict[str, Any]:
    """
    Parse a YAML configuration file into a dictionary.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary keyed by attribute names from the config file
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Return empty dict if file is empty or None
    return config_dict if config_dict is not None else {}


def parse_live_config() -> Dict[str, Any]:
    """
    Parse the default live_config.yaml file.
    
    Returns:
        Dictionary keyed by attribute names from live_config.yaml
    """
    # Get the path relative to this file's location
    config_path = Path(__file__).parent.parent / "live_config.yaml"
    return parse_config(str(config_path))

