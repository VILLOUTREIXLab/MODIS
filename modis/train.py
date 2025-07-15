import os
import pathlib
import time

import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from modis.utils.data import summarize_dataset

def train(
    train_datasets: list[torch.utils.data.DataLoader],
    config: DictConfig,
    data_summary: bool = True
):
    # Instantiate dataloaders
    train_dataloaders = [DataLoader(dataset, batch_size=config.batch_size, drop_last=True, shuffle=True)
                         for dataset in train_datasets]
    
    # Variables
    num_modalities = len(config.modalities)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    init_epoch = 0

    # Paths
    save_path = pathlib.Path("./saved")
    checkpoint_path = save_path / "checkpoints" / config.dataset_name / config.model_name
    logs_path = save_path / "logs" / config.dataset_name / config.model_name

    if data_summary:
        print("==> Summary of train datasets")
        modality_names = [m.name for m in config.modalities]
        summarize_dataset(train_dataloaders, modality_names=modality_names)

