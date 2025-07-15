# import sys
# import os
# script_dir = (os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..',))
# sys.path.insert(0, script_dir)

import modis
from modis.utils.config import load_config

config = load_config('config/intersim/intersim.yaml')

modis.train(config)