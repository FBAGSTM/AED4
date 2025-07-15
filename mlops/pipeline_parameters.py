from sagemaker.workflow.parameters import ParameterFloat
from typing import Union, Dict, Any
import yaml
import os 

def get_dataset_name(aed_mz_folder_name: str) -> str:
    """
    Extract the dataset name from a YAML configuration file.

    Args:
        aed_mz_folder_name (str): The folder name in ModelZoo (mz) for AED use case.

    Returns:
        str: The name of the dataset specified in the YAML file.
    """
    try:
        current_dir = os.getcwd()
        aed_mz_dir = os.path.join(current_dir, "AED", aed_mz_folder_name)
        user_training_config = os.path.join(aed_mz_dir, "scripts", "training", "user_config.yaml")
        with open(user_training_config, "r", encoding="utf-8") as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config.get("dataset", {}).get("name", "AED_Dataset")

    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        print(f"Error: {e}")
        return "AED_Dataset"


def get_pipeline_parameters(default_threshold: Union[float, None] = None) -> Dict[str, Any]:
    """
    Return a dict that contains specific parameters pipeline for AED usecase

    Args:
        default_threshold (Union[float, None]): A threshold value that pipeline parameters need to be above. If None, a default value will be used.

    Returns:
        Dict[str, Any]: A dictionary containing the pipeline parameters tailored for the AED use case.
    """
    AED_MZ_FOLDER_NAME = "audio_event_detection"
    CLIP_PARAM_NAME = "q_clip_level_acc_threshold"
    PATCH_PARAM_NAME = "q_patch_level_acc_threshold"
    config = {
        "use_case_modelzoo_folder": AED_MZ_FOLDER_NAME,
        "processing_step_name": f"Preprocess_{get_dataset_name(AED_MZ_FOLDER_NAME)}",
        "train_operation_mode": "chain_tqe",
        "validation_parameters": [
            {
                "parameter": {
                    "object": ParameterFloat(name = CLIP_PARAM_NAME, default_value = default_threshold if default_threshold else 0.5),
                    "fail_step_msg": "Execution failed due to clip level acc <",
                    "json_path": "multiclass_classification_metrics.clip_acc.value"
                },
                "metric_definition": {
                    "Name": "clip:accuracy",
                    "Regex": "Clip-level accuracy on test set : ([0-9\\.]+)",
                }
            },
            {
                "parameter": {
                    "object": ParameterFloat(name = PATCH_PARAM_NAME, default_value = default_threshold if default_threshold else 0.5),
                    "fail_step_msg": "Execution failed due to patch level acc <",
                    "json_path": "multiclass_classification_metrics.patch_acc.value"
                },
                "metric_definition": {
                    "Name": "patch:accuracy",
                    "Regex": "Patch-level accuracy on test set : ([0-9\\.]+)",
                }
            }
        ]
    }
    return config