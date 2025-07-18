import sys
import os
script_dir = (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',))
sys.path.insert(0, script_dir)

import torch
from torch.utils.data import random_split

import modis
from modis.utils.data import SemiSupervisedDataset

from src.intersim_dataset import get_datasets

random_seed = 1234

config_file = 'config/intersim.yaml'

train_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'paired',
    split = 'train',
    data_path = './data',
    include_sample_ids = False
)

# Train/validation split
train_datasets, val_datasets = list(zip(*[random_split(
    train_dataset, 
    [0.8, 0.2],  # Train and validation fractions
    generator=torch.Generator().manual_seed(random_seed)
) for train_dataset in train_datasets]))

# Generate partially labeled dataset
from modis.utils.config import load_config
config = load_config(config_file)
if config.training_mode == 'semisupervised':
    # Unsupervised dataset
    # train_datasets = [SemiSupervisedDataset(dataset, labeled_ratio=0, random_seed=random_seed) for dataset in train_datasets]

    # Semisupervised dataset
    train_datasets = [SemiSupervisedDataset(dataset, labeled_ratio=0.00, random_seed=random_seed) for dataset in train_datasets]
    # train_datasets = [SemiSupervisedDataset(dataset, class_samples=[1, 1, 1, 1, 1], random_seed=random_seed) for dataset in train_datasets]

# Train model
# checkpoint_dir = modis.train(config_file, train_datasets, summarize_datasets=True, report_plots=True)
checkpoint_dir = modis.train(
    config_file,
    train_datasets,
    val_datasets,
    show_dataset_summary=True,
    run_evaluation=True,
    generate_plots=True
)


# Evaluation on test dataset

from modis.utils.data import get_dataloaders
from modis.utils.config import load_config
from modis.utils.utils import evaluate_model
from modis.model import MODIS

print(f"\n==> Evaluating model on test dataset")

test_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'test',
    data_path = './data',
    include_sample_ids = False
)

config = load_config(config_file)
model = MODIS(config)
model.load_from_checkpoint(checkpoint_dir / "checkpoint_best.pth")

test_dataloaders = get_dataloaders(test_datasets, batch_size=config.batch_size, drop_last=False, shuffle=False)
metrics  = evaluate_model(model, test_dataloaders)

for k,v in metrics.items():
    print(f"{k}: {v:.4f}")
