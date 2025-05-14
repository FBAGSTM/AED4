from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterFloat
)

def get_pipeline_parameters() -> dict:
    """
    Return a dict that contains specific parameters pipeline for AED usecase
    """
    config = {
        "use_case_folder": "audio_event_detection",
        "processing_step_name": "PreprocessFSD50K",
        "modelzoo_version": "v1",
        "validation_parameters": [
            {
                "parameter": {
                    "object": ParameterFloat(name = "q_clip_level_acc_threshold", default_value = 0.5),
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
                    "object": ParameterFloat(name = "q_patch_level_acc_threshold", default_value = 0.5),
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