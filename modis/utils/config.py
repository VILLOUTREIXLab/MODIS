import sys

import torch
import omegaconf

def validate_config(config: omegaconf.DictConfig) -> None:
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

    if type(config.beta) == omegaconf.listconfig.ListConfig:
        assert len(config.beta) == len(config.modalities), "The 'beta' parameter in the configuration must be a list whose length matches the number of modalities."
    elif type(config.beta) != float:
        raise TypeError("The 'beta' parameter in the configuration must be a float or a list.")

def load_config(config_file: str) -> omegaconf.DictConfig:
    """Load a configuration from config file"""
    config = omegaconf.OmegaConf.load(config_file)
    validate_config(config)
    return config