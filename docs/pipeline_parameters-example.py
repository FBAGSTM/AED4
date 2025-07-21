from sagemaker.workflow.parameters import ParameterFloat
from typing import Union, Dict, Any


def get_pipeline_parameters(
    default_threshold: Union[float, None] = None,
) -> Dict[str, Any]:
    AED_MZ_FOLDER_NAME = "audio_event_detection"
    CLIP_PARAM_NAME = "q_clip_level_acc_threshold"
    config = {
        "use_case_modelzoo_folder": AED_MZ_FOLDER_NAME,
        "processing_step_name": f"Preprocess_AED_Dataset",
        "modelzoo_version": "v3",
        "train_operation_mode": "chain_tqe",
        "eval_path_method": "swap",
        "validation_parameters": [
            {
                "parameter": {
                    "object": ParameterFloat(
                        name=CLIP_PARAM_NAME,
                        default_value=default_threshold if default_threshold else 0.5,
                    ),
                    "fail_step_msg": "Execution failed due to clip level acc <",
                    "json_path": "multiclass_classification_metrics.clip_acc.value",
                },
                "metric_definition": {
                    "Name": "clip:accuracy",
                    "Regex": "Clip-level accuracy on test set : ([0-9\\.]+)",
                },
            }
        ],
    }
    return config
