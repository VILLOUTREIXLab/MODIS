import modis
from src.generate_dataset import get_datasets

# Load the dataset
train_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'train',
    data_dir = './data',
    include_sample_ids = False
)

# Train model
config_file = 'config/supervised.yaml'
modis.train(config_file=config_file, train_datasets=train_datasets)
