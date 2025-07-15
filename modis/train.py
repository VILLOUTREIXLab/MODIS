import os
import pathlib

import torch
from omegaconf import DictConfig

def train(
    config: DictConfig
):
    num_modalities = len(config.modalities)
    save_path = pathlib.Path("./saved")
    checkpoint_path = save_path / "checkpoints" 

    print(checkpoint_path)
    # os.path.join(config.checkpoint_folder, config.model_name)
