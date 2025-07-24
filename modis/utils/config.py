import sys
import pathlib
from itertools import product

import torch
from omegaconf import OmegaConf, DictConfig, ListConfig

def flatten_omegaconf(config, parent_key='') -> dict:
    """Return the config as flatten dict"""
    items = {}
    if isinstance(config, DictConfig):
        for k, v in config.items():
            new_key = f"{parent_key}.{k}" if parent_key else str(k)
            if OmegaConf.is_config(v):
                items.update(flatten_omegaconf(v, new_key))
            else:
                items[new_key] = v
    elif isinstance(config, ListConfig):
        for idx, v in enumerate(config):
            new_key = f"{parent_key}.{idx}" if parent_key else str(idx)
            if OmegaConf.is_config(v):
                items.update(flatten_omegaconf(v, new_key))
            else:
                items[new_key] = v
    return items

def validate_config(config: DictConfig) -> None:
    """Validate a configuration"""
    error = None
    
    # Required fields validation
    required_fields = [
        'dataset_name', 'model_name', 'latent_size', 'modalities', 
        'training_mode', 'batch_size', 'num_epochs', 'generators_lr', 
        'discriminator_lr', 'beta', 'beta1', 'lambda_r', 'device',
        'save_checkpoint_latest', 'save_checkpoint_best'
    ]
    
    for field in required_fields:
        if field not in config:
            error = f'Missing required field: {field}'
            break
    
    # Training mode validation
    if error is None and config.training_mode not in ['semisupervised', 'supervised']:
        error = f'Invalid training mode: {config.training_mode}'
    
    # Device validation
    elif error is None and config.device not in ['auto', 'cuda', 'cpu']:
        error = f'Invalid device: {config.device}'
    
    # Numeric validations
    elif error is None and config.latent_size <= 0:
        error = f'Invalid latent_size: {config.latent_size}. Must be positive integer.'
    
    elif error is None and config.batch_size <= 0:
        error = f'Invalid batch_size: {config.batch_size}. Must be positive integer.'
    
    elif error is None and config.num_epochs <= 0:
        error = f'Invalid num_epochs: {config.num_epochs}. Must be positive integer.'
    
    elif error is None and config.generators_lr <= 0:
        error = f'Invalid generators_lr: {config.generators_lr}. Must be positive.'
    
    elif error is None and config.discriminator_lr <= 0:
        error = f'Invalid discriminator_lr: {config.discriminator_lr}. Must be positive.'
    
    elif error is None and not (0 <= config.beta1 <= 1):
        error = f'Invalid beta1: {config.beta1}. Must be between 0 and 1.'
    
    elif error is None and config.lambda_r < 1:
        error = f'Invalid lambda_r: {config.lambda_r}. Must be greater than 1.'
    
    # Modalities validation
    elif error is None:
        if not config.modalities or len(config.modalities) < 2:
            error = 'At least two modality must be specified'
        else:
            for i, modality in enumerate(config.modalities):
                if 'name' not in modality:
                    error = f'Modality {i} missing required field: name'
                    break
                elif 'input_size' not in modality:
                    error = f'Modality {i} ({modality.name}) missing required field: input_size'
                    break
                elif modality.input_size <= 0:
                    error = f'Invalid input_size for modality {modality.name}: {modality.input_size}. Must be positive integer.'
                    break
                elif 'encoder_ratios' in modality:
                    if not isinstance(modality.encoder_ratios, (list, ListConfig)):
                        error = f'Invalid encoder_ratios for modality {modality.name}: must be a list'
                        break
                    elif len(modality.encoder_ratios) == 0:
                        error = f'Invalid encoder_ratios for modality {modality.name}: list cannot be empty'
                        break
                    elif any(ratio <= 0 for ratio in modality.encoder_ratios):
                        error = f'Invalid encoder_ratios for modality {modality.name}: all ratios must be positive'
                        break
    
    # Supervised mode specific validation
    if error is None and config.training_mode == 'supervised':
        if 'num_classes' not in config:
            error = 'num_classes is required for supervised training mode'
        elif config.num_classes <= 0:
            error = f'Invalid num_classes: {config.num_classes}. Must be positive integer.'
    
    # Boolean fields validation
    if error is None:
        boolean_fields = ['save_checkpoint_latest', 'save_checkpoint_best']
        for field in boolean_fields:
            if not isinstance(config[field], bool):
                error = f'Invalid {field}: {config[field]}. Must be boolean (true/false).'
                break

    if error is None:
        if type(config.beta) == ListConfig and len(config.beta) != len(config.modalities):
            error = "Invalid beta parameter. The length of the list must match the number of modalities."
        elif type(config.beta) != float:
            error = "Invalid beta parameter. Must be a float or a list."
    
    if error is not None:
        print(f'[-] Configuration error: {error}')
        sys.exit(1)
    
    # Auto device detection
    if config.device == 'auto':
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

def set_param(config, param_name, param_value):
    if '.' in param_name:
        if OmegaConf.select(config, param_name) is None:
            raise KeyError(f"Parameter '{param_name}' does not exist in configuration")
        OmegaConf.update(config, param_name, param_value)
    else:
        if param_name not in config:
            raise KeyError(f"Parameter '{param_name}' does not exist in configuration")
        config[param_name] = param_value
        
def generate_grid(base_config_file: pathlib.Path, params: dict):
    """
    Generate a grid of configurations from a base config and parameter variations.
    
    Args:
        base_config_file: Path to the base configuration file
        params: Dictionary of parameters to vary. Keys can be:
                - Simple keys: 'latent_size'
                - Nested keys: 'modalities.0.encoder_ratios'
                Values should be lists of possible param values.
    
    Returns:
        List of OmegaConf DictConfig objects, each representing one parameter combination
    
    Raises:
        FileNotFoundError: If base config file doesn't exist
        ValueError: If parameters are not provided or the config file fails to load
    """
    if not base_config_file.exists():
        raise FileNotFoundError(f"Base config file not found: {base_config_file}")
    
    if not params:
        raise ValueError("Parameters dictionary cannot be empty")

    try:
        base_config = load_config(base_config_file)
    except Exception as e:
        raise ValueError(f"Failed to load config from {base_config_file}: {e}")

    grid_params = list(params.keys())
    param_values = list(params.values())
    combinations = list(product(*param_values))

    grid = []
    for combo in combinations:
        # Start with a copy of base_config
        new_config = OmegaConf.create(OmegaConf.to_yaml(base_config))

        # Apply grid parameters
        for param_name, param_value in zip(grid_params, combo):
            set_param(new_config, param_name, param_value)

        validate_config(new_config)
        grid.append(new_config)

    return grid

def load_config(config_file: str) -> DictConfig:
    """Load a configuration from config file"""
    config = OmegaConf.load(config_file)
    validate_config(config)
    return config