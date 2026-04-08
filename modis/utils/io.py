"""
I/O utilities for MODIS.

This module provides helpers for loading checkpoints, training logs, and
configuration files from disk.
"""
import json
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from modis.utils.config import validate_config


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load a model checkpoint from disk.

    Args:
        checkpoint_file (pathlib.Path): Path to the ``.pth`` checkpoint file.

    Returns:
        dict: Checkpoint dictionary containing at least ``'model_state'``,
        ``'optimizer_state'``, ``'epoch'``, ``'best_epoch'``,
        ``'best_loss'``, ``'val_acc'``, and ``'timestamp'``.

    Raises:
        FileNotFoundError: If ``checkpoint_file`` does not exist.
    """
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} doesn't exist.")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    return checkpoint


def load_log(checkpoint_file: Path) -> list:
    """Load a training log from a JSON file.

    Args:
        checkpoint_file (pathlib.Path): Path to the JSON log file.

    Returns:
        list[dict]: List of per-epoch metric dictionaries, in chronological
        order.

    Raises:
        FileNotFoundError: If ``checkpoint_file`` does not exist.
    """
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Log file {checkpoint_file} doesn't exist.")
    with open(checkpoint_file, 'r', encoding='utf-8') as file:
        log = json.load(file)
    return log


def load_config(config_file: str) -> DictConfig:
    """Load and validate a MODIS configuration from a YAML file.

    Args:
        config_file (str or pathlib.Path): Path to the YAML configuration
            file.

    Returns:
        omegaconf.DictConfig: The loaded and validated configuration object.

    Raises:
        SystemExit: If the configuration fails validation (see
            :func:`~modis.utils.config.validate_config`).
    """
    config = OmegaConf.load(config_file)
    validate_config(config)
    return config