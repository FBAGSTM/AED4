# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import sys
import numpy as np
from hydra.core.hydra_config import HydraConfig
import hydra
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from omegaconf import DictConfig
import mlflow
import argparse
import logging
from typing import Optional
from clearml import Task
from clearml.backend_config.defs import get_active_config_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.evaluation import gen_load_val
from src.preprocessing import preprocess
from src.utils import get_config, AED_CUSTOM_OBJECTS
from src.training import train
from src.evaluation import evaluate
from src.quantization import quantize
from src.prediction import predict
from deployment import deploy



def chain_qd(cfg: DictConfig = None, float_model_path: str = None, train_ds: tf.data.Dataset = None,
             quantization_ds: tf.data.Dataset = None) -> None: 
    """
    Runs the chain_qd pipeline, including quantization,  and deployment

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        float_model_path (str): float model path to evaluate and quantize. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stm32ai.on_cloud:
        _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)

    if quantization_ds:
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds,
                                        float_model_path=float_model_path)
    elif train_ds:
        print('[INFO] : Quantization dataset is not provided! Using the training set to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=train_ds,
                                        float_model_path=float_model_path)
    else:
        print('[INFO] : Neither quantization dataset nor training set are provided! Using fake data to quantize the model. '
              'The model performances will not be accurate.')
        quantized_model_path = quantize(cfg=cfg, fake=True)
    print('[INFO] : Quantization complete.')

    deploy(cfg=cfg, model_path_to_deploy=quantized_model_path, credentials=credentials)
    print('[INFO] : Deployment complete.')


def chain_eqeb(cfg: DictConfig = None, float_model_path: str = None, train_ds: tf.data.Dataset = None,
               valid_ds: tf.data.Dataset = None, valid_clip_labels: np.ndarray = None,
               quantization_ds: tf.data.Dataset = None, test_ds: tf.data.Dataset = None,
               test_clip_labels: np.ndarray = None, multi_label: bool = None) -> None:
    """
    Runs the chain_eqeb pipeline, including evaluation of the float model,  quantization , evaluation of
    the quantized model and benchmarking

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        float_model_path (str): float model path to evaluate and quantize. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        valid_clip_labels : np.ndarray, Clip labels for the validation dataset
        test_ds (tf.data.Dataset): Test dataset. Defaults to None.
        test_clip_labels : np.ndarray, Clip labels for the test dataset.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None
        multi_label : bool, set to True if the dataset is multi_label

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stm32ai.on_cloud:
        _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)

    # Check for batch size both in training & general section
    if cfg.training:
        if cfg.training.batch_size:
            batch_size = cfg.training.batch_size
    elif cfg.general.batch_size:
        batch_size = cfg.general.batch_size
    else:
        batch_size = 32

    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds,
             clip_labels=test_clip_labels, multi_label=multi_label, 
             model_path_to_evaluate=float_model_path, batch_size=batch_size,
             name_ds='test_set')
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds,
            clip_labels=valid_clip_labels, multi_label=multi_label, 
            model_path_to_evaluate=float_model_path, batch_size=batch_size,
            name_ds='validation_set')
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)
    if quantization_ds:
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds,
                                        float_model_path=float_model_path)
    elif train_ds:
        print('[INFO] : Quantization dataset is not provided! Using the training set to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=train_ds,
                                        float_model_path=float_model_path)
    else:
        print('[INFO] : No quantization or training dataset provided. Quantizing using fake data')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=train_ds,
                                    float_model_path=float_model_path, fake=True)
    print('[INFO] : Quantization complete.')
    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds,
             clip_labels=test_clip_labels, multi_label=multi_label, 
             model_path_to_evaluate=quantized_model_path, batch_size=batch_size,
             name_ds='test_set')
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds,
            clip_labels=valid_clip_labels, multi_label=multi_label, 
            model_path_to_evaluate=quantized_model_path, batch_size=batch_size,
            name_ds='validation_set')
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)

    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials, custom_objects=AED_CUSTOM_OBJECTS)
    print('[INFO] : Benchmarking complete.')


def chain_qb(cfg: DictConfig = None, float_model_path: str = None, train_ds: tf.data.Dataset = None,
             quantization_ds: tf.data.Dataset = None) -> None:
    """
    Runs the chain_qb pipeline, including quantization and benchmarking.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        float_model_path (str): float_model_path to quantize. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None

    Returns:
        None
    """

    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stm32ai.on_cloud:
        _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)

    if quantization_ds:
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds,
                                        float_model_path=float_model_path)
    elif train_ds:
        print('[INFO] : Quantization dataset is not provided! Using the training set to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=train_ds,
                                        float_model_path=float_model_path)
    else:
        print('[INFO] : Neither quantization dataset nor training set are provided! Using fake data to quantize the model. '
              'The model performances will not be accurate.')
        quantized_model_path = quantize(cfg=cfg, fake=True)
    print('[INFO] : Quantization complete.')

    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials, custom_objects=AED_CUSTOM_OBJECTS)
    print('[INFO] : Benchmarking complete.')


def chain_eqe(cfg: DictConfig = None, float_model_path: str = None, train_ds: tf.data.Dataset = None,
              valid_ds: tf.data.Dataset = None, valid_clip_labels: np.ndarray = None,
              quantization_ds: tf.data.Dataset = None, test_ds: tf.data.Dataset = None,
              test_clip_labels: np.ndarray = None, multi_label: bool = None) -> None:
    """
    Runs the chain_eqe pipeline, including evaluation of a float model,  quantization and evaluation of
    the quantized model

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        float_model_path (str): float model path to evaluate and quantize. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        valid_clip_labels : np.ndarray, Clip labels for the validation dataset
        test_ds (tf.data.Dataset): Test dataset. Defaults to None.
        test_clip_labels : np.ndarray, Clip labels for the test dataset.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None
        multi_label : bool, set to True if the dataset is multi_label

    Returns:
        None
    """

    # Check for batch size both in training & general section
    if cfg.training:
        if cfg.training.batch_size:
            batch_size = cfg.training.batch_size
    elif cfg.general.batch_size:
        batch_size = cfg.general.batch_size
    else:
        batch_size = 32

    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds,
             clip_labels=test_clip_labels, multi_label=multi_label, 
             model_path_to_evaluate=float_model_path, batch_size=batch_size,
             name_ds='test_set')
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds,
            clip_labels=valid_clip_labels, multi_label=multi_label, 
            model_path_to_evaluate=float_model_path, batch_size=batch_size,
            name_ds='validation_set')
    print('[INFO] : Evaluation complete.')
    if quantization_ds:
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds,
                                        float_model_path=float_model_path)
    elif train_ds:
        print('[INFO] : Quantization dataset is not provided! Using the training set to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=train_ds,
                                        float_model_path=float_model_path)
    else:
        print('[INFO] : No quantization or training dataset provided. Quantizing using fake data')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=train_ds,
                                    float_model_path=float_model_path, fake=True)
    print('[INFO] : Quantization complete.')
    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds,
             clip_labels=test_clip_labels, multi_label=multi_label, 
             model_path_to_evaluate=quantized_model_path, batch_size=batch_size,
             name_ds='test_set')
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds,
            clip_labels=valid_clip_labels, multi_label=multi_label, 
            model_path_to_evaluate=quantized_model_path, batch_size=batch_size,
            name_ds='validation_set')
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)


def chain_tbqeb(cfg: DictConfig = None, train_ds: tf.data.Dataset = None,
                valid_ds: tf.data.Dataset = None, valid_clip_labels: np.ndarray = None,
                quantization_ds: tf.data.Dataset = None, test_ds: tf.data.Dataset = None,
                test_clip_labels: np.ndarray = None, multi_label: bool = None) -> None:
    """
    Runs the chain_tbqeb pipeline, including training,  benchmarking , quantization, evaluation and benchmarking.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        valid_ds (tf.data.Dataset): Validation dataset. Defaults to None.
        valid_clip_labels : np.ndarray, Clip labels for the validation dataset
        test_ds (tf.data.Dataset): Test dataset. Defaults to None.
        test_clip_labels : np.ndarray, Clip labels for the test dataset.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None
        multi_label : bool, set to True if the dataset is multi_label

    Returns:
        None
    """

    # Check for batch size both in training & general section
    if cfg.training:
        if cfg.training.batch_size:
            batch_size = cfg.training.batch_size
    elif cfg.general.batch_size:
        batch_size = cfg.general.batch_size
    else:
        batch_size = 32


    # Connect to STM32Cube.AI Developer Cloud
    credentials = None
    if cfg.tools.stm32ai.on_cloud:
        _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)

    if test_ds:
        trained_model_path = train(cfg=cfg, train_ds=train_ds, valid_ds=valid_ds,
                                   val_clip_labels=valid_clip_labels, test_ds=test_ds,
                                   test_clip_labels=test_clip_labels)
    else:
        trained_model_path = train(cfg=cfg, train_ds=train_ds, valid_ds=valid_ds,
                                   val_clip_labels=valid_clip_labels)
    print('[INFO] : Training complete.')
    benchmark(cfg=cfg, model_path_to_benchmark=trained_model_path, credentials=credentials, custom_objects=AED_CUSTOM_OBJECTS)
    print('[INFO] : benchmarking complete.')
    if quantization_ds:
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds,
                                        float_model_path=trained_model_path)
    else:
        print('[INFO] : Quantization dataset is not provided! Using the training set to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=train_ds,
                                        float_model_path=trained_model_path)
    print('[INFO] : Quantization complete.')
    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds,
            clip_labels=test_clip_labels, multi_label=multi_label, 
            model_path_to_evaluate=quantized_model_path, batch_size=batch_size,
            name_ds='test_set')
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds,
            clip_labels=valid_clip_labels, multi_label=multi_label, 
            model_path_to_evaluate=quantized_model_path, batch_size=batch_size,
            name_ds='validation_set')
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)
    benchmark(cfg=cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials, custom_objects=AED_CUSTOM_OBJECTS)
    print('[INFO] : Benchmarking complete.')


def chain_tqe(cfg: DictConfig = None, train_ds: tf.data.Dataset = None,
              valid_ds: tf.data.Dataset = None, valid_clip_labels: np.ndarray = None,
              quantization_ds: tf.data.Dataset = None, test_ds: tf.data.Dataset = None,
              test_clip_labels: np.ndarray = None, multi_label: bool = None) -> None:
    """
    Runs the chain_tqe pipeline, including training,  quantization and evaluation.

    Args:
        cfg (DictConfig): Configuration dictionary. Defaults to None.
        train_ds (tf.data.Dataset): Training dataset. Defaults to None.
        valid_clip_labels : np.ndarray, Clip labels for the validation dataset
        test_ds (tf.data.Dataset): Test dataset. Defaults to None.
        test_clip_labels : np.ndarray, Clip labels for the test dataset.
        quantization_ds:(tf.data.Dataset): quantization dataset. Defaults to None
        multi_label : bool, set to True if the dataset is multi_label

    Returns:
        None
    """

    # Check for batch size both in training & general section
    if cfg.training:
        if cfg.training.batch_size:
            batch_size = cfg.training.batch_size
    elif cfg.general.batch_size:
        batch_size = cfg.general.batch_size
    else:
        batch_size = 32

    if test_ds:
        trained_model_path = train(cfg=cfg, train_ds=train_ds, valid_ds=valid_ds,
                                   val_clip_labels=valid_clip_labels, test_ds=test_ds,
                                   test_clip_labels=test_clip_labels)
    else:
        trained_model_path = train(cfg=cfg, train_ds=train_ds, valid_ds=valid_ds,
                                   val_clip_labels=valid_clip_labels)
    print('[INFO] : Training complete.')
    if quantization_ds:
        print('[INFO] : Using the quantization dataset to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=quantization_ds,
                                        float_model_path=trained_model_path)
    else:
        print('[INFO] : Quantization dataset is not provided! Using the training set to quantize the model.')
        quantized_model_path = quantize(cfg=cfg, quantization_ds=train_ds,
                                        float_model_path=trained_model_path)
    print('[INFO] : Quantization complete.')
    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds,
            clip_labels=test_clip_labels, multi_label=multi_label, 
            model_path_to_evaluate=quantized_model_path, batch_size=batch_size,
            name_ds='test_set')
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds,
            clip_labels=valid_clip_labels, multi_label=multi_label, 
            model_path_to_evaluate=quantized_model_path, batch_size=batch_size,
            name_ds='validation_set')
    print('[INFO] : Evaluation complete.')
    display_figures(cfg)

def process_mode(mode: str = None, configs: DictConfig = None, train_ds: tf.data.Dataset = None,
                 valid_ds: tf.data.Dataset = None, valid_clip_labels: np.ndarray = None,
                 quantization_ds: tf.data.Dataset = None, test_ds: tf.data.Dataset = None,
                 test_clip_labels: np.ndarray = None, multi_label: bool = None,
                 float_model_path: Optional[str] = None, fake: Optional[bool] = False) -> None:
    """
    Process the selected mode of operation.

    Args:
        mode (str): The selected mode of operation. Must be one of 'train', 'evaluate', or 'predict'.
        configs (DictConfig): The configuration object.
        train_ds (tf.data.Dataset): The training dataset. Required if mode is 'train'.
        valid_ds (tf.data.Dataset): The validation dataset. Required if mode is 'train' or 'evaluate'.
        valid_clip_labels : np.ndarray, Clip labels for the validation dataset
        test_ds (tf.data.Dataset): The test dataset. Required if mode is 'evaluate' or 'predict'.
        test_clip_labels : np.ndarray, Clip labels for the test dataset.
        quantization_ds(tf.data.Dataset): The quantization dataset.
        multi_label : bool, set to True if the dataset is multi_label
        float_model_path(str, optional): Model path . Defaults to None
        fake (bool, optional): Whether to use fake data for representative dataset generation. Defaults to False.
    Returns:
        None
    Raises:
        ValueError: If an invalid mode is selected
    """

    mlflow.log_param("model_path", configs.general.model_path)
    # Check for batch size both in training & general section
    if configs.training:
        if configs.training.batch_size:
            batch_size = configs.training.batch_size
    elif configs.general.batch_size:
        batch_size = configs.general.batch_size
    else:
        batch_size = 32

    # Check the selected mode and perform the corresponding operation
    # logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(configs.output_dir, f'operation_mode: {mode}')
    if mode == 'training':
        if test_ds:
            train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds,
                  val_clip_labels=valid_clip_labels, test_ds=test_ds,
                  test_clip_labels=test_clip_labels)
        else:
             train(cfg=configs, train_ds=train_ds, valid_ds=valid_ds,
                   val_clip_labels=valid_clip_labels)
        display_figures(configs)
        print('[INFO] : Training complete.')

    elif mode == 'evaluation':
        # =============================================================================
        # NOTE: This mode has been customized for our AWS-GitHub setup
        # to allow SageMaker to collect evaluation metrics required for model validation.
        import boto3
        from sagemaker.experiments.run import load_run
        from sagemaker.session import Session

        if "REGION" not in os.environ:
            raise EnvironmentError(
                f"Expected environment variable 'REGION' not found. "
                f"Available environment variables: {list(os.environ.keys())}"
            )
        session = Session(boto3.session.Session(region_name=os.environ.get("REGION")))

        # Generates the model to be loaded on the stm32n6 device using stedgeai core,
        # then loads it and validates in on the device if required.
        gen_load_val(cfg=configs)
        # Launches evaluation on the target through the model zoo evaluation service
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        sagemaker_report_dict = None
        if test_ds:
            sagemaker_report_dict = evaluate(cfg=configs, eval_ds=test_ds,
                clip_labels=test_clip_labels, multi_label=multi_label, 
                model_path_to_evaluate=None, batch_size=batch_size,
                name_ds='test_set')
        else:
            sagemaker_report_dict = evaluate(cfg=configs, eval_ds=valid_ds,
                clip_labels=valid_clip_labels, multi_label=multi_label, 
                model_path_to_evaluate=None, batch_size=batch_size,
                name_ds='validation_set')
        
        with load_run(experiment_name=os.environ["EXPERIMENT_NAME"], run_name=os.environ["RUN_NAME"], sagemaker_session=session) as run:
            run.log_metric(name="patch_acc", value=sagemaker_report_dict["multiclass_classification_metrics"]["patch_acc"]["value"])
            run.log_metric(name="clip_acc", value=sagemaker_report_dict["multiclass_classification_metrics"]["clip_acc"]["value"])

        #   AWS SageMaker Evaluation Report Generation
        #
        #   In AWS SageMaker pipelines, a PropertyFile is used to pass outputs from one
        #   processing step to subsequent steps, such as for conditional logic or reporting.
        #
        #   The following code generates an `evaluation.json` file containing evaluation
        #   metrics. This file is saved in the designated output directory so that
        #   SageMaker can access and use these metrics for model validation, conditional
        #   branching, or automated approval workflows.
        #
        #   For more details, see the AWS SageMaker documentation on PropertyFile:
        #   https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
        import pathlib
        import json

        print("[INFO] : Generating evaluation report file for SageMaker ...")
        output_dir = "/opt/ml/processing/evaluation"
        try:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            evaluation_path = f"{output_dir}/evaluation.json"
            with open(evaluation_path, "w") as f:
                json.dump(sagemaker_report_dict, f)
            print("[INFO] : Successfully created evaluation.json")
        except Exception as e:
            raise Exception(f"Failed to write evaluation.json: {e}")

        # =============================================================================

        display_figures(configs)
        print('[INFO] : Evaluation complete.')
    elif mode == 'deployment':
        deploy(cfg=configs)
        print('[INFO] : Deployment complete.')
        if configs.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')
    elif mode == 'quantization':
        if quantization_ds:
            input_ds = quantization_ds
            fake = False
            print('[INFO] : Using the quantization dataset to quantize the model.')
        elif train_ds:
            print('[INFO] : Quantization dataset is not provided! Using the training set to quantize the model.')
            input_ds = train_ds
            fake = False
        else:
            input_ds = None
            fake = True
            print(
                '[INFO] : Neither quantization dataset or training set are provided! Using fake data to quantize the model. '
                'The model performances will not be accurate.')
        quantize(cfg=configs, quantization_ds=input_ds, fake=fake)
        print('[INFO] : Quantization complete.')
    elif mode == 'prediction':
        predict(cfg=configs)
        print('[INFO] : Prediction complete.')
    elif mode == 'benchmarking':
        benchmark(cfg=configs, custom_objects=AED_CUSTOM_OBJECTS)
        print('[INFO] : Benchmark complete.')
    elif mode == 'chain_tbqeb':
        chain_tbqeb(cfg=configs, train_ds=train_ds, valid_ds=valid_ds,
                    valid_clip_labels=valid_clip_labels, quantization_ds=quantization_ds,
                    test_ds=test_ds, test_clip_labels=test_clip_labels, multi_label=multi_label)
        print('[INFO] : chain_tbqeb complete.')
    elif mode == 'chain_tqe':
        chain_tqe(cfg=configs, train_ds=train_ds, valid_ds=valid_ds,
                    valid_clip_labels=valid_clip_labels, quantization_ds=quantization_ds,
                    test_ds=test_ds, test_clip_labels=test_clip_labels, multi_label=multi_label)
        print('[INFO] : chain_tqe complete.')
    elif mode == 'chain_eqe':
        chain_eqe(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, float_model_path=float_model_path,
                    valid_clip_labels=valid_clip_labels, quantization_ds=quantization_ds,
                    test_ds=test_ds, test_clip_labels=test_clip_labels, multi_label=multi_label)
        print('[INFO] : chain_eqe complete.')
    elif mode == 'chain_qb':
        chain_qb(cfg=configs, float_model_path=float_model_path, train_ds=train_ds,
                 quantization_ds=quantization_ds)
        print('[INFO] : chain_qb complete.')
    elif mode == 'chain_eqeb':
        chain_eqeb(cfg=configs, train_ds=train_ds, valid_ds=valid_ds, float_model_path=float_model_path,
                    valid_clip_labels=valid_clip_labels, quantization_ds=quantization_ds,
                    test_ds=test_ds, test_clip_labels=test_clip_labels, multi_label=multi_label)
        print('[INFO] : chain_eqeb complete.')
    elif mode == 'chain_qd':
        chain_qd(cfg=configs, float_model_path=float_model_path, train_ds=train_ds,
                 quantization_ds=quantization_ds)
        print('[INFO] : chain_qd complete.')
    # Raise an error if an invalid mode is selected
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(configs.output_dir)
    if mode in ['benchmarking', 'chain_qb', 'chain_eqeb', 'chain_tbqeb']:
        mlflow.log_param("stm32ai_version", configs.tools.stm32ai.version)
        mlflow.log_param("target", configs.benchmarking.board)
    # logging the completion of the chain
    log_to_file(configs.output_dir, f'operation finished: {mode}')

    # ClearML - Example how to get task's context anywhere in the file.
    # Checks if there's a valid ClearML configuration file
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(configs)


# -----------------------------------------------------------------------------
# Custom utility for AWS-GitHub architecture:
# This function validates and adjusts the dataset-specific configuration paths
# for the FSD50K dataset according to the current operation mode.
# It ensures that all required paths are present and correctly formatted for
# SageMaker/CodeBuild (e.g., using the right mount prefixes), and disables
# them when not needed (e.g., in deployment mode).
# This logic is specific to our pipeline and is not present in the original Model Zoo.
def handle_fsd50k_config(cfg: DictConfig) -> None:
    """
    Validate and adjust fsd50k dataset-specific configuration paths
    according to the operation mode.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Raises:
        ValueError: If required keys are missing or invalid.
    """
    dataset_specific = cfg.get("dataset_specific")
    if not isinstance(dataset_specific, DictConfig):
        raise ValueError(
            f"[ERROR] Missing or invalid 'dataset_specific' key: {dataset_specific}"
        )

    fsd50k = dataset_specific.get("fsd50k")
    if not isinstance(fsd50k, DictConfig):
        raise ValueError(f"[ERROR] Missing or invalid 'fsd50k' key: {fsd50k}")

    keys_to_check: List[str] = [
        "csv_folder",
        "dev_audio_folder",
        "eval_audio_folder",
        "audioset_ontology_path",
    ]

    prefix_path_train = "/opt/ml/input/data/train/"
    prefix_code_train = "/opt/ml/code"
    entry_prefixes = [prefix_path_train, prefix_code_train]

    operation_mode = getattr(cfg, "operation_mode", None)
    if operation_mode is None:
        raise ValueError(
            "[ERROR] 'operation_mode' is not defined in the configuration."
        )
    print(
        f"[INFO] Configuring fsd50k keys based on operation mode detected: '{operation_mode}'"
    )

    for key in keys_to_check:
        value = fsd50k.get(key)
        if not (
            isinstance(value, str) and any(value.startswith(p) for p in entry_prefixes)
        ):
            raise ValueError(
                f"[ERROR] dataset_specific.fsd50k.{key} value '{value}' "
                f"does not exist or does not start with any of the expected prefixes {entry_prefixes}. "
                "Please check your pipeline configuration."
            )

        if operation_mode in {"evaluation", "chain_eqe", "benchmarking", "chain_eqeb"}:
            if value.startswith(prefix_path_train):
                value = value.replace(prefix_path_train, "/opt/ml/processing/input/", 1)
            elif value.startswith(prefix_code_train):
                value = value.replace(
                    prefix_code_train, "/opt/ml/processing/input/code/", 1
                )
            fsd50k[key] = value

        elif operation_mode == "deployment":
            fsd50k[key] = None
    print("[INFO] Sucessfully done configuring 'fsd50k' yaml file.")
# -----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """

   # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
            print(f"[INFO] : Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")
            
    # -------------------------------------------------------------------------
    # Custom logic for AWS-GitHub integration (not present in the original Model Zoo):
    # Special handling for the FSD50K dataset to ensure all file paths are
    # compatible with SageMaker/CodeBuild mount points and operation modes.
    import re
    from omegaconf import OmegaConf 

    dataset_cfg = cfg.get("dataset")
    if not (dataset_cfg and isinstance(dataset_cfg, DictConfig)):
        raise ValueError("[ERROR] 'dataset' section missing or invalid in config")
    dataset_name = dataset_cfg.get("name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("[ERROR] 'name' section missing or invalid in config")

    if dataset_name.lower().strip() == "fsd50k":
        print(
            "[INFO] Dataset identified as 'fsd50k'. Performing special configuration ..."
        )
        pattern = re.compile(r"_path$")
        for key in dataset_cfg.keys():
            key_clean = key.strip().lower()
            if pattern.search(key_clean) and isinstance(dataset_cfg[key], str):
                dataset_cfg[key] = None
        handle_fsd50k_config(cfg)

    print("[DEBUG] Hydra Configuration received: ")
    print(OmegaConf.to_yaml(cfg))
    # -------------------------------------------------------------------------

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir
    mlflow_ini(cfg)

    # Checks if there's a valid ClearML configuration file
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        # ClearML - Initializing ClearML's Task object.
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='aed_modelzoo_task')
        # ClearML - Optional yaml logging 
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # Extract the mode from the command-line arguments
    mode = cfg.operation_mode
    valid_modes = ['training', 'evaluation', 'chain_tbqeb', 'chain_tqe']
    if mode in valid_modes:
        # Load datasets
        train_ds, valid_ds, valid_clip_labels, quantization_ds, test_ds, test_clip_labels = preprocess(cfg=cfg)
        # Assert if dataset is multilabel
        multi_label = cfg.dataset.multi_label
        # Process the selected mode
        process_mode(mode=mode, 
                     configs=cfg, 
                     train_ds=train_ds, 
                     valid_ds=valid_ds,
                     valid_clip_labels=valid_clip_labels, 
                     quantization_ds=quantization_ds,
                     test_ds=test_ds, 
                     test_clip_labels=test_clip_labels, 
                     multi_label=multi_label)
    elif mode == 'quantization':
        if cfg.dataset.training_audio_path or cfg.dataset.quantization_audio_path:
            train_ds, valid_ds, valid_clip_labels, quantization_ds, test_ds, test_clip_labels = preprocess(cfg=cfg)
            # Process the selected mode
            multi_label = cfg.dataset.multi_label
            process_mode(mode=mode, 
                         configs=cfg, 
                         train_ds=train_ds, 
                         valid_ds=valid_ds,
                         valid_clip_labels=valid_clip_labels, 
                         quantization_ds=quantization_ds,
                         test_ds=test_ds, 
                         test_clip_labels=test_clip_labels,
                         multi_label=multi_label)
        else:
            process_mode(mode=mode, 
                         configs=cfg, 
                         fake=True)
    else:
        if mode in ['chain_eqe', 'chain_qb', 'chain_eqeb', 'chain_qd']:
            if (cfg.dataset.training_audio_path or cfg.dataset.quantization_audio_path or
                 cfg.dataset.validation_audio_path or cfg.dataset.test_audio_path):
                train_ds, valid_ds, valid_clip_labels, quantization_ds, test_ds, test_clip_labels = preprocess(cfg=cfg)
            else:
                raise TypeError("No dataset provided")
            
            if quantization_ds or train_ds:
                fake=False
            else:
                fake=True
            process_mode(mode=mode, 
                         configs=cfg, 
                         train_ds=train_ds, 
                         valid_ds=valid_ds,
                         valid_clip_labels=valid_clip_labels,
                         quantization_ds=quantization_ds,
                         test_ds=test_ds, 
                         test_clip_labels=test_clip_labels,
                         multi_label=cfg.dataset.multi_label,
                         float_model_path=cfg.general.model_path,
                         fake=fake)
        else:
            # Process the selected mode
            process_mode(mode=mode, 
                         configs=cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')
    # add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
