import sys
import os
script_dir = (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',))
sys.path.insert(0, script_dir)

import torch
from torch.utils.data import random_split

import modis
from modis.utils.data import SemiSupervisedDataset

from src.intersim_dataset import get_datasets

config_file = 'config/intersim/intersim.yaml'

train_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'train',
    data_path = './data',
    include_sample_ids = False
)

# Train/validation split
train_datasets, val_datasets = list(zip(*[random_split(
    train_dataset, 
    [0.8, 0.2],  # Train and validation fractions
    generator=torch.Generator().manual_seed(1234)
) for train_dataset in train_datasets]))

from modis.utils.config import load_config
config = load_config(config_file)
if config.training_mode == 'semisupervised':
    # Fully unsupervised dataset
    # train_datasets = [SemiSupervisedDataset(dataset, labeled_ratio=0, random_seed=1234) for dataset in train_datasets]

    # Fully semisupervised dataset
    train_datasets = [SemiSupervisedDataset(dataset, labeled_ratio=0.1, random_seed=1234) for dataset in train_datasets]

# Train model
checkpoint_path = modis.train(config_file, train_datasets, val_datasets, summarize_datasets=True, report_plots=True)
