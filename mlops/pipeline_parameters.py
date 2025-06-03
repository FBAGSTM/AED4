from sagemaker.workflow.parameters import (
    ParameterFloat
)

from typing import Union, Dict, Any

def get_pipeline_parameters(dataset_name: str = "FSD50k", default_threshold: Union[float, None] = None) -> Dict[str, Any]:
    """
    Return a dict that contains specific parameters pipeline for AED usecase

    Args:
        dataset_name (str): The name of the dataset to be used. Defaults to "FSD50k".
        default_threshold (Union[float, None]): A threshold value that pipeline parameters need to be above. If None, a default value will be used.

    Returns:
        Dict[str, Any]: A dictionary containing the pipeline parameters tailored for the AED use case.
    """
    CLIP_PARAM_NAME = "q_clip_level_acc_threshold"
    PATCH_PARAM_NAME = "q_patch_level_acc_threshold"
    config = {
        "use_case_modelzoo_folder": "audio_event_detection",
        "processing_step_name": f"Preprocess{dataset_name}",
        "modelzoo_version": "v1",
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