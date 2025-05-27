# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import sys,os,logging
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import hydra
from omegaconf import DictConfig

# BUG: Code changed here
from pathlib import Path
# Define paths using pathlib
base_dir = Path(__file__).resolve().parent
os.chdir(base_dir)
model_path = Path('/opt/ml/model')
train_data_path = Path('/opt/ml/input/data/train')
utils_path = base_dir / '..' / 'utils'
models_path = utils_path / 'models'
common_path = base_dir / '..' / '..' / '..' / 'common'
# Add paths to sys.path if not already present
for path in [model_path, train_data_path, utils_path.resolve(), models_path.resolve(), common_path.resolve()]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from utils import get_config, mlflow_ini, setup_seed, train

@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg : DictConfig) -> None:
    print(cfg)
    #initilize configuration & mlflow
    configs = get_config(cfg)
    mlflow_ini(configs)

    # Set all seeds
    setup_seed(42)

    #train the model
    train(configs)

if __name__ == "__main__":
    main()