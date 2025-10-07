import modis
from modis.utils.data import PartiallyLabeledDataset, get_dataloaders, random_split
from modis.utils.evaluation import evaluate_model

from src.datasets import get_datasets

random_seed = 1234
config = modis.load_config('config/semisupervised.yaml')

# Load the dataset
train_datasets = get_datasets(
    dataset_name = 'toy_dataset',
    split = 'train',
    include_ids = False,
    data_dir = './data'
)

# Generate train/validation data splits
train_datasets, val_datasets = random_split(train_datasets, [0.8, 0.2], random_seed)

# Generate artificially a partially labeled dataset
train_datasets = [PartiallyLabeledDataset(dataset, labeled_samples_ratio=0.05, random_seed=random_seed)
                  for dataset in train_datasets]

# Train model
checkpoint_dir = modis.train(
    config=config,
    train_datasets=train_datasets,
    val_datasets=val_datasets
)

# Evaluation on test dataset

print(f"\n==> Evaluating model on test dataset")

test_datasets = get_datasets(
    dataset_name = 'toy_dataset',
    split = 'test',
    include_ids = False,
    data_dir = './data'
)

model = modis.Model(config)
model.load_from_checkpoint(checkpoint_dir / "checkpoint_best.pth")

test_dataloaders = get_dataloaders(test_datasets, batch_size=config.batch_size, drop_last=False, shuffle=False)
metrics = evaluate_model(model, test_dataloaders)

for k,v in metrics.items():
    if isinstance(v, list):
        print(f"{k}: {[f"{val:.4f}" for val in v]}")
    else:
        print(f"{k}: {v:.4f}")
