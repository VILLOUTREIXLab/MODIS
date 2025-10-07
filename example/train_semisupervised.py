import modis
from modis.utils.data import PartiallyLabeledDataset, random_split

from src.datasets import get_datasets

# Load the dataset
train_datasets = get_datasets(
    dataset_name = 'toy_dataset',
    split = 'train',
    include_ids = False,
    data_dir = './data'
)

# Generate a validation set
train_datasets, val_datasets = random_split(train_datasets, [0.8, 0.2], paired=False)

# Generate artificially a partially labeled dataset (only 10% supervision)
train_datasets = [
    PartiallyLabeledDataset(dataset, labeled_samples_ratio=0.10)
    for dataset in train_datasets
]

# Train model
config = modis.load_config('config/semisupervised.yaml')
modis.train(
    config=config,
    train_datasets=train_datasets,
    val_datasets=val_datasets
)
