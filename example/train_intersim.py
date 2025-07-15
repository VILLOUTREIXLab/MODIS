import sys
import os
script_dir = (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',))
sys.path.insert(0, script_dir)

import modis
from modis.utils.config import load_config
from modis.utils.data import get_dataloaders
from src.intersim_dataset import get_datasets

config = load_config('config/intersim/intersim.yaml')

train_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'train',
    data_path = './data',
    include_sample_ids = False
)

modis.train(train_datasets, config, data_summary=True)