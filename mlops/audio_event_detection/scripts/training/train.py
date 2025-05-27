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
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath('/opt/ml/model'))
sys.path.append(os.path.abspath('/opt/ml/input/data/train'))

sys.path.append(os.path.abspath('../utils'))
sys.path.append(os.path.abspath('../utils/models'))

# BUG: Code changed here
common_directory = 'pipelines' / 'stm' / 'stm32ai-modelzoo-v1' / 'common'
abs_common_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', common_directory))
if abs_common_directory not in sys.path:
    print(f"Module path: {abs_common_directory} added to sys path.")
    sys.path.append(abs_common_directory)

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