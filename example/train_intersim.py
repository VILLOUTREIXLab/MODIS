import sys
import os
script_dir = (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',))
sys.path.insert(0, script_dir)

import modis
from src.intersim_dataset import get_datasets


train_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'train',
    data_path = './data',
    include_sample_ids = False
)

modis.train(
    train_datasets,
    config_file='config/intersim/intersim.yaml',
    data_summary=True
)