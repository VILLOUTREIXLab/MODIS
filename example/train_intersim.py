import sys
import os
script_dir = (os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',))
sys.path.insert(0, script_dir)

import modis
from modis.utils.plots import checkpoint_report_plots

from src.intersim_dataset import get_datasets

config_file = 'config/intersim/intersim.yaml'

train_datasets = get_datasets(
    dataset_name = 'intersim_2_delta',
    pairing = 'unpaired',
    split = 'train',
    data_path = './data',
    include_sample_ids = False
)

checkpoint_path = modis.train(train_datasets, config_file=config_file, data_summary=True)

checkpoint_report_plots(
    checkpoint_path = checkpoint_path,
    config_file = config_file,
    datasets = train_datasets,
    use_best = True,
    num_samples = None
)