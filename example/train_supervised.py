import modis
from src.datasets import get_datasets

# Load the dataset
train_datasets = get_datasets(
    dataset_name = 'toy_dataset',
    split = 'train',
    include_ids = False,
    data_dir = './data'
)

# Train model
config = modis.load_config('config/supervised.yaml')
modis.train(config=config, train_datasets=train_datasets)
