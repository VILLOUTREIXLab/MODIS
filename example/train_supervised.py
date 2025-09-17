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
config = modis.load_config('config/supervised.yaml')
modis.train(config=config, train_datasets=train_datasets)
