import sys

import torch
from omegaconf import OmegaConf, DictConfig

def validate_config(config: DictConfig) -> None:
    """Validate a configuration"""
    error = None
    if config.training_mode not in ['semisupervised', 'supervised']:
        error = f'Invalid training mode: {config.training_mode}'
    elif config.device not in ['auto', 'cuda', 'cpu']:
        error = f'Invalid device: {config.device}'

    if error is not None:
        print(f'[-] Configuration error: {error}')
        sys.exit(1)

    if config.device == 'auto':
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(config_file: str) -> DictConfig:
    """Load a configuration from config file"""
    config = OmegaConf.load(config_file)
    # validate_config(config)
    return config