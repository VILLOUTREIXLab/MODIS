import modis
import torch
from torch.utils.data import random_split
from modis.utils.data import PartiallyLabeledDataset
from src.generate_dataset import get_datasets

random_seed = 1234

# Load the dataset
train_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'train',
    data_dir = './data',
    include_sample_ids = False
)

# Generate train/validation data splits
train_datasets, val_datasets = list(zip(*[random_split(
    dataset = train_dataset,
    lengths = [0.8, 0.2],
    generator = torch.Generator().manual_seed(random_seed)
) for train_dataset in train_datasets]))

# Generate artificially a partially labeled dataset
train_datasets = [PartiallyLabeledDataset(dataset, labeled_ratio=0.2, random_seed=random_seed)
                  for dataset in train_datasets]

# Train model
modis.train(
    config_file='config/semisupervised.yaml',
    train_datasets=train_datasets,
    val_datasets=val_datasets
)
