import sys
import os
script_dir = (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',))
sys.path.insert(0, script_dir)

import torch
from torch.utils.data import random_split

import modis
from modis.utils.plots import checkpoint_report_plots

from src.intersim_dataset import get_datasets

config_file = 'config/intersim/intersim.yaml'

train_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'train',
    data_path = './data',
    include_sample_ids = False
)

train_datasets, val_datasets = list(zip(*[random_split(
    train_dataset, 
    [0.8, 0.2],  # Train and validation fractions
    generator=torch.Generator().manual_seed(1234)
) for train_dataset in train_datasets]))

checkpoint_path = modis.train(config_file, train_datasets, val_datasets, data_summary=True)

checkpoint_report_plots(
    checkpoint_path = checkpoint_path,
    config_file = config_file,
    datasets = train_datasets,
    use_best = True,
    num_samples = None
)