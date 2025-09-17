import json
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from modis.utils.config import validate_config

def load_checkpoint(checkpoint_file: Path) -> dict:
    """Loads a checkpoint from a specified file path.

    Args:
        checkpoint_file: Path to the checkpoint file.

    Returns:
        A dictionary containing the checkpoint data.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} doesn't exist.")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    return checkpoint

def load_log(checkpoint_file: Path) -> list:
    """Loads a log from a specified file path.

    Args:
        checkpoint_file: Path to the log file.

    Returns:
        A list containing the log data.

    Raises:
        FileNotFoundError: If the log file does not exist.
    """
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Log file {checkpoint_file} doesn't exist.")
    with open(checkpoint_file, 'r', encoding='utf-8') as file:
        log = json.load(file)
    return log

def load_config(config_file: str) -> DictConfig:
    """Loads a configuration from a config file.

    Args:
        config_file: Path to the configuration file.

    Returns:
        A DictConfig object containing the loaded and validated configuration.
    """
    config = OmegaConf.load(config_file)
    validate_config(config)
    return config