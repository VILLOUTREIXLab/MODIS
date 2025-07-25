import modis
from modis.utils.data import PartiallyLabeledDataset, get_dataloaders, random_split
from modis.utils.io import load_config
from modis.utils.evaluation import evaluate_model

from src.generate_dataset import get_datasets

random_seed = 1234
config_file = 'config/semisupervised.yaml'

# Load the dataset
train_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'train',
    data_dir = './data',
    include_sample_ids = False
)

# Generate train/validation data splits
train_datasets, val_datasets = random_split(train_datasets, [0.8, 0.2], random_seed)

# Generate artificially a partially labeled dataset
train_datasets = [PartiallyLabeledDataset(dataset, labeled_ratio=0.05, random_seed=random_seed)
                  for dataset in train_datasets]

# Train model
checkpoint_dir = modis.train(
    config_file=config_file,
    train_datasets=train_datasets,
    val_datasets=val_datasets
)

# Evaluation on test dataset

print(f"\n==> Evaluating model on test dataset")

test_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'test',
    data_dir = './data',
    include_sample_ids = False
)

config = load_config(config_file)
model = modis.Model(config)
model.load_from_checkpoint(checkpoint_dir / "checkpoint_best.pth")

test_dataloaders = get_dataloaders(test_datasets, batch_size=config.batch_size, drop_last=False, shuffle=False)
metrics  = evaluate_model(model, test_dataloaders)

for k,v in metrics.items():
    print(f"{k}: {v:.4f}")
